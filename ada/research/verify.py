"""``ada verify``: check that the assets on disk still follow from ``study.yml`` and the data.

It checks four things and reports each under its own heading:

1. Inputs: the data file is there, with the checksum in ``study.yml`` and the
   one the build recorded.
2. Out of date: each asset was built from the current ``study.yml``, ADA
   version and custom scripts, and nothing ``study.yml`` asks for is missing.
3. Edited: no file has changed since the build recorded its checksum.
4. Rebuild: ``study.yml``, the data and the custom scripts, copied alone into a
   scratch folder as a fresh checkout would hold them, rebuild to the same
   assets. Numbers in CSV files may differ by a tolerance, figures are
   compared by their source data, Word files by their text, and every other
   text file exactly.
"""

from __future__ import annotations

import difflib
import os
import re
import shutil
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import CHANGES, MANIFEST, TEXT_SUFFIXES, join, name, read_manifest, sections_sentence
from .study import Study, StudyError, load, sha256_of

#: Default tolerance for numbers in CSV files. As in ``numpy.isclose``, two numbers match when
#: |rebuilt − disk| ≤ ATOL + RTOL × |disk|.
RTOL = 1e-9
ATOL = 1e-12
#: Differences listed per file; the rest are counted.
SHOWN = 5
SECTIONS = {
    "inputs": "Inputs",
    "stale": "Out of date",
    "edited": "Edited since the build",
    "rebuild": "Rebuild in a scratch folder",
}
IMAGE_SUFFIXES = {".png", ".pdf", ".svg", ".tif", ".tiff", ".eps", ".jpg", ".jpeg"}
TABLE_SUFFIXES = {".csv": ",", ".tsv": "\t"}
#: Lines that hold a release date by convention, by file name. They are left out of the comparison,
#: and the report says so whenever one differs.
VOLATILE = {
    "CITATION.cff": ("date-released", re.compile(r"^date-released:.*$")),
    ".zenodo.json": ("publication_date", re.compile(r'^\s*"publication_date":.*$')),
}
_MARKS = {"ok": "ok  ", "fail": "FAIL", "note": "note"}
#: Characters of a value shown in a message; longer values are cut around the first difference.
_WIDTH = 60


@dataclass(frozen=True)
class Line:
    """One finding. ``status`` is "ok", "fail" or "note"; ``detail`` is shown indented under it."""

    status: str
    text: str
    detail: tuple[str, ...] = ()


@dataclass
class Report:
    study: str = "study.yml"
    #: The findings under each heading of :data:`SECTIONS`: inputs, stale, edited, rebuild.
    lines: dict[str, list[Line]] = field(default_factory=lambda: {key: [] for key in SECTIONS})
    summary: str = ""

    @property
    def ok(self) -> bool:
        return not self.failures()

    def failures(self, section: str | None = None) -> list[Line]:
        keys = [section] if section else list(self.lines)
        return [line for key in keys for line in self.lines[key] if line.status == "fail"]

    def add(self, section: str, status: str, text: str, detail=()) -> None:
        self.lines[section].append(Line(status, text, tuple(detail)))

    def text(self) -> str:
        out = [f"ada verify: {self.study}", ""]
        for key, heading in SECTIONS.items():
            if not self.lines[key]:
                continue
            out.append(heading)
            for line in self.lines[key]:
                out += textwrap.wrap(
                    line.text,
                    width=100,
                    initial_indent=f"  {_MARKS[line.status]}  ",
                    subsequent_indent=" " * 8,
                    break_on_hyphens=False,
                )
                out += [f"{' ' * 10}{item}" for item in line.detail]
            out.append("")
        out.append(self.summary)
        return "\n".join(out)

    def to_dict(self) -> dict:
        """The same findings as :meth:`text`, as one JSON-serialisable object.

        ``study``, ``ok`` and ``summary`` come first, then each section of :data:`SECTIONS` as a list
        of findings, in the order the report prints them. A section the report leaves out is an empty
        list, so a reader can ask for any of them. Two runs on the same study give the same object:
        it holds no timestamps, and every path in it is the one the finding named.
        """
        return {
            "study": self.study,
            "ok": self.ok,
            "summary": self.summary,
            **{
                key: [
                    {"status": line.status, "text": line.text, "detail": list(line.detail)}
                    for line in lines
                ]
                for key, lines in self.lines.items()
            },
        }


def verify(
    study: str | Path = "study.yml",
    *,
    rtol: float = RTOL,
    atol: float = ATOL,
    out: str | Path | None = None,
) -> Report:
    """Check the assets in ``out`` (the study's output folder by default) against ``study`` and a rebuild."""
    loaded = load(study)
    out = Path(out).resolve() if out else loaded.out_dir
    report = Report(study=_shown(loaded.path))
    record = read_manifest(out)
    can_rebuild = _check_inputs(loaded, record, report)
    if record is None:
        report.add(
            "inputs",
            "fail",
            f"There is no build in {_shown(out)} to verify (no {MANIFEST}). Run `ada build` first.",
        )
        report.summary = "Not verified: nothing has been built yet."
        return report
    stale, scripts = _stale(loaded, record)
    deleted = _check_edits(out, record, report)
    rebuilt = None
    if can_rebuild:
        rebuilt = _rebuild(
            loaded, out, record, report, rtol=rtol, atol=atol, deleted=deleted, stale=set(stale)
        )
    else:
        report.add("rebuild", "note", "Not rebuilt, because an input above is missing or has changed.")
    _report_stale(loaded, record, stale, scripts, rebuilt, report)
    report.summary = _summary(report, record)
    return report


def _shown(path: Path) -> str:
    """``path`` relative to the working folder when it is inside it, so messages stay short."""
    try:
        return path.relative_to(Path.cwd()).as_posix() or "."
    except ValueError:
        return str(path)


def _check_inputs(study: Study, record: dict | None, report: Report) -> bool:
    """Report on the data file and custom scripts; return whether a rebuild can run."""
    can_rebuild = True
    built = ((record or {}).get("inputs") or [{}])[0]
    path = study.data_file
    if not path.is_file():
        report.add(
            "inputs",
            "fail",
            f"The data file {study.data_path} is not there (looked for {path}). Restore it, or correct "
            "`data.path` in study.yml.",
        )
        can_rebuild = False
    else:
        digest = sha256_of(path)
        if study.data_sha256 and digest != study.data_sha256:
            report.add(
                "inputs",
                "fail",
                f"{study.data_path} has changed: its sha256 is now {digest[:12]}…, but study.yml records "
                f"{study.data_sha256[:12]}…. Restore the original file or, if the change is intended, put "
                "the new checksum in `data.sha256` and run `ada build`.",
            )
            can_rebuild = False
        elif built.get("sha256") and digest != built["sha256"]:
            report.add(
                "inputs",
                "fail",
                f"{study.data_path} is not the file the assets were built from: its sha256 is "
                f"{digest[:12]}…, and the build used {built['sha256'][:12]}…. Run `ada build`.",
            )
        else:
            matches = join(
                (["study.yml"] if study.data_sha256 else []) + (["the build"] if built.get("sha256") else [])
            )
            report.add(
                "inputs",
                "ok",
                f"{study.data_path} "
                + (f"matches {matches}" if matches else "is there")
                + f" (sha256 {digest[:12]}…).",
            )
        if not study.data_sha256:
            report.add(
                "inputs",
                "note",
                "study.yml records no `data.sha256`, so `ada build` would not notice a changed data file. "
                f"Add `sha256: {digest}` under `data:`.",
            )
    if Path(study.data_path).is_absolute():
        report.add(
            "inputs",
            "note",
            "`data.path` is an absolute path, which a copy of this folder on another computer will not have. "
            "Keep the data beside study.yml and give its path relative to it.",
        )
    for entry in study.custom:
        if not (study.root / entry.script).is_file():
            report.add(
                "inputs", "fail", f"The custom script {entry.script} listed in study.yml is not there."
            )
            can_rebuild = False
    return can_rebuild


def _stale(study: Study, record: dict) -> tuple[dict[str, str], list[str]]:
    """Assets built from parts of study.yml or custom scripts that have changed since, with the reason.

    Also returns the custom scripts that changed. A different ADA version is reported separately,
    since a new version often makes the same files.
    """
    assets = record.get("assets", {})
    then = record.get("study_sections")
    reasons: dict[str, list[str]] = {}
    for ident, entry in assets.items():
        depends = entry.get("depends_on", [])
        if study.section_hash(*depends) != entry.get("config_sha256"):
            changed = (
                None if then is None else [key for key in depends if study.section_hash(key) != then.get(key)]
            )
            reasons.setdefault(ident, []).append(sections_sentence(changed, depends))
    built = {item["script"]: item["sha256"] for item in record.get("custom", [])}
    scripts = []
    for entry in study.custom:
        path = study.root / entry.script
        if entry.script not in built or not path.is_file() or sha256_of(path) == built[entry.script]:
            continue
        scripts.append(entry.script)
        for ident, item in assets.items():
            if item.get("code", "").startswith(f"{entry.script}:"):
                reasons.setdefault(ident, []).append(f"{entry.script} changed")
    return {ident: " and ".join(items) for ident, items in reasons.items()}, scripts


def _report_stale(
    study: Study,
    record: dict,
    stale: dict[str, str],
    scripts: list[str],
    rebuilt: dict | None,
    report: Report,
) -> None:
    from .. import __version__  # noqa: PLC0415

    assets = record.get("assets", {})
    groups: dict[str, list[str]] = {}
    for ident, reason in stale.items():
        groups.setdefault(reason, []).append(name(ident, assets[ident]))
    for reason, names in groups.items():
        report.add("stale", "fail", f"Out of date because {reason} since the build: {', '.join(names)}.")
    for script in scripts:
        if not any(f"{script} changed" in reason for reason in stale.values()):
            report.add("stale", "fail", f"{script} has changed since the build. Run `ada build`.")
    found = bool(stale or scripts)
    if record.get("ada") != __version__:
        report.add(
            "stale",
            "fail",
            f"The assets were built by ADA {record.get('ada')}, and this is ADA {__version__}. Run "
            "`ada build` to rebuild them with this version.",
        )
        found = True
    if rebuilt is not None:
        implied = rebuilt.get("assets", {})
        new = [name(ident, entry) for ident, entry in implied.items() if ident not in assets]
        gone = [name(ident, entry) for ident, entry in assets.items() if ident not in implied]
        if new:
            report.add(
                "stale",
                "fail",
                f"Missing: study.yml now asks for {', '.join(new)}, which "
                + ("has" if len(new) == 1 else "have")
                + " not been built. Run `ada build`.",
            )
        if gone:
            report.add(
                "stale",
                "fail",
                f"In the build, but study.yml no longer makes {'it' if len(gone) == 1 else 'them'}: "
                f"{', '.join(gone)}. Run `ada build`.",
            )
        found = found or bool(new or gone)
    if not found:
        scripts_note = " and custom scripts" if study.custom else ""
        report.add(
            "stale",
            "ok",
            f"All {len(assets)} assets were built from the current study.yml{scripts_note} by this "
            f"version of ADA ({__version__}).",
        )
    if rebuilt is None:
        report.add("stale", "note", "Could not check for assets study.yml asks for but the build lacks.")
    versions = _version_changes(record)
    if versions:
        report.add(
            "stale",
            "note",
            "Versions differ from the build: " + ", ".join(versions) + ". The rebuild shows whether any "
            "number changed.",
        )


def _version_changes(record: dict) -> list[str]:
    import platform  # noqa: PLC0415

    from .provenance import package_versions  # noqa: PLC0415

    changes = []
    if record.get("python") and record["python"] != platform.python_version():
        changes.append(f"Python {record['python']} → {platform.python_version()}")
    now = package_versions()
    for package, version in record.get("packages", {}).items():
        if package in now and now[package] != version:
            changes.append(f"{package} {version} → {now[package]}")
    return changes


def _check_edits(out: Path, record: dict, report: Report) -> set[str]:
    """Report recorded files edited or deleted since the build; return the deleted ones."""
    edited, deleted, total = [], [], 0
    for entry in record.get("assets", {}).values():
        for relative, digest in entry.get("files", {}).items():
            total += 1
            path = out / relative
            if not path.is_file():
                deleted.append(relative)
            elif sha256_of(path) != digest:
                edited.append(relative)
    if edited:
        report.add(
            "edited",
            "fail",
            f"Changed since the build: {', '.join(edited)}. The next `ada build` replaces "
            + ("it" if len(edited) == 1 else "them")
            + "; make lasting changes in study.yml or a custom script.",
        )
    if deleted:
        report.add("edited", "fail", f"Deleted since the build: {', '.join(deleted)}.")
    if not edited and not deleted:
        report.add("edited", "ok", f"None of the {total} files recorded in {MANIFEST} has changed.")
    return set(deleted)


def _copy_study(study: Study, scratch: Path) -> Study:
    """Copy study.yml, the data and the custom scripts into ``scratch``, keeping their relative paths."""
    sources = [study.path]
    if not Path(study.data_path).is_absolute():
        sources.append(Path(os.path.normpath(study.root / study.data_path)))
    sources += [Path(os.path.normpath(study.root / entry.script)) for entry in study.custom]
    anchor = Path(os.path.commonpath([str(source.parent) for source in sources]))
    for source in sources:
        target = scratch / source.relative_to(anchor)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return load(scratch / study.path.relative_to(anchor))


def _rebuild(
    study: Study,
    out: Path,
    record: dict,
    report: Report,
    *,
    rtol: float,
    atol: float,
    deleted: set[str],
    stale: set[str],
) -> dict | None:
    """Build a copy of the study in a scratch folder and compare every file with ``out``."""
    from .build import build  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="ada-verify-", ignore_cleanup_errors=True) as folder:
        scratch = Path(folder).resolve()
        copy = _copy_study(study, scratch)
        target = copy.out_dir if copy.out_dir.is_relative_to(scratch) else scratch / "paper-assets"
        try:
            built = build(copy, out=target)
        except Exception as error:  # noqa: BLE001
            message = " ".join(error.problems) if isinstance(error, StudyError) else str(error).strip()
            message = message or type(error).__name__
            report.add(
                "rebuild",
                "fail",
                f"The rebuild failed: {message.rstrip('.')}. It uses only study.yml, the data file and the "
                "custom scripts study.yml lists, as a fresh copy of the study would.",
            )
            return None
        rebuilt = read_manifest(built.out) or {}
        comparison = _Comparison(out, built.out, record, rebuilt, report, rtol=rtol, atol=atol)
        comparison.run(deleted=deleted, stale=stale)
        report.add(
            *comparison.verdict(join([study.path.name, study.data_path, *[e.script for e in study.custom]]))
        )
        return rebuilt


@dataclass
class _Comparison:
    """The file-by-file comparison of the assets on disk with a rebuild."""

    out: Path
    rebuilt_out: Path
    record: dict
    rebuilt: dict
    report: Report
    rtol: float
    atol: float
    counts: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(("table", "text", "word", "figure", "other"), 0)
    )
    failed: int = 0
    #: Files of a kind ADA cannot read that differ in bytes: reported, but not counted as differences.
    unread: list[str] = field(default_factory=list)

    def run(self, *, deleted: set[str], stale: set[str]) -> None:
        disk_assets, new_assets = self.record.get("assets", {}), self.rebuilt.get("assets", {})
        owner = {path: ident for ident, entry in disk_assets.items() for path in entry.get("files", {})}
        owner.update(
            {
                path: ident
                for ident, entry in new_assets.items()
                for path in entry.get("files", {})
                if path not in owner
            }
        )
        made = sorted(
            path.relative_to(self.rebuilt_out).as_posix()
            for path in self.rebuilt_out.rglob("*")
            if path.is_file()
        )
        # Files the build writes outside any asset, such as README.md and figures/captions.md.
        others = [path for path in made if path not in owner and path not in (CHANGES, MANIFEST)]
        expected, images, sources = [], [], {}
        for relative in [*owner, *others]:
            if relative in deleted:
                continue
            ident = owner.get(relative)
            disk, fresh = self.out / relative, self.rebuilt_out / relative
            if not fresh.is_file() or not disk.is_file():
                self._absent(relative, ident, disk_assets, new_assets, fresh.is_file())
                continue
            suffix = disk.suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                images.append((relative, ident, disk.read_bytes() == fresh.read_bytes()))
                continue
            differences, total = self._compare(relative, disk, fresh, suffix)
            if relative.endswith(".source" + suffix):
                sources[ident] = not differences
            if not differences:
                continue
            if ident in stale or (ident is None and stale):
                expected.append(relative)
                continue
            self.failed += 1
            more = total - len(differences)
            self.report.add(
                "rebuild",
                "fail",
                f"{relative} differs from the rebuild ({total:,} difference{'s' if total != 1 else ''}):",
                [*differences, *([f"… and {more:,} more"] if more > 0 else [])],
            )
        if expected:
            self.failed += 1
            self.report.add(
                "rebuild",
                "fail",
                "Differ from the rebuild, as expected while their assets are out of date (run `ada build`): "
                + ", ".join(expected)
                + ".",
            )
        if self.unread:
            self.report.add(
                "rebuild",
                "note",
                "These differ in bytes, and ADA cannot compare what is in them: "
                + ", ".join(self.unread)
                + ".",
            )
        self._images(images, sources)
        self._strays(set(owner) | set(made))

    def _absent(self, relative: str, ident, disk_assets: dict, new_assets: dict, rebuilt: bool) -> None:
        """A file on one side only. A whole asset missing on one side is reported under Out of date."""
        if not rebuilt and ident in new_assets:
            self.failed += 1
            self.report.add(
                "rebuild", "fail", f"{relative} is in the build, but the rebuild does not make it."
            )
        elif rebuilt and (ident is None or ident in disk_assets):
            self.failed += 1
            self.report.add(
                "rebuild", "fail", f"{relative}: the rebuild makes it, but it is not in {_shown(self.out)}."
            )

    def _compare(self, relative: str, disk: Path, fresh: Path, suffix: str) -> tuple[list[str], int]:
        if suffix in TABLE_SUFFIXES:
            self.counts["table"] += 1
            return compare_table(
                disk, fresh, rtol=self.rtol, atol=self.atol, separator=TABLE_SUFFIXES[suffix]
            )
        if suffix == ".docx":
            self.counts["word"] += 1
            return compare_word(disk, fresh)
        if suffix not in TEXT_SUFFIXES and not (_is_text(disk) and _is_text(fresh)):
            if disk.read_bytes() == fresh.read_bytes():
                self.counts["other"] += 1
            else:
                self.unread.append(relative)
            return [], 0
        self.counts["text"] += 1
        differences, total, ignored = compare_text(disk, fresh)
        if ignored:
            self.report.add(
                "rebuild", "note", f"Ignored {ignored} in {relative}, which holds the date it was written."
            )
        return differences, total

    def _images(self, images: list[tuple[str, str | None, bool]], sources: dict) -> None:
        """Images match when their figure's source data match; byte differences are only noted."""
        rendering, unsourced, with_data = [], [], []
        for relative, ident, same in images:
            self.counts["figure"] += 1
            if same:
                continue
            if ident not in sources:
                unsourced.append(relative)
            elif sources[ident]:
                rendering.append(relative)
            else:
                with_data.append(relative)
        if rendering:
            self.report.add(
                "rebuild",
                "note",
                "These images differ in bytes, but their source data match, so they show the same data "
                "(fonts or library versions differ): " + ", ".join(rendering) + ".",
            )
        if with_data:
            self.report.add(
                "rebuild",
                "note",
                "These images differ along with their source data: " + ", ".join(with_data) + ".",
            )
        if unsourced:
            self.report.add(
                "rebuild",
                "note",
                "These images differ in bytes and have no source data to compare: "
                + ", ".join(unsourced)
                + ".",
            )

    def _strays(self, known: set[str]) -> None:
        """Note files in the folders the build owns that it did not make: the next build deletes them."""
        from .build import OWNED  # noqa: PLC0415

        strays = sorted(
            path.relative_to(self.out).as_posix()
            for folder in OWNED
            if (self.out / folder).is_dir()
            for path in (self.out / folder).rglob("*")
            if path.is_file() and path.relative_to(self.out).as_posix() not in known
        )
        if strays:
            self.report.add(
                "edited",
                "note",
                "Not made by ADA, and deleted by the next `ada build`, which empties the folders it owns: "
                + ", ".join(strays)
                + ". Keep your own files outside them.",
            )

    def verdict(self, copied: str) -> tuple[str, str, str]:
        if self.failed:
            return ("rebuild", "note", f"The rebuild used {copied}, alone in a scratch folder.")
        counts = self.counts
        parts = [
            f"{counts['table']} CSV files number by number (rtol {self.rtol:g}, atol {self.atol:g})"
            if counts["table"]
            else "",
            f"{counts['text']} text files exactly" if counts["text"] else "",
            f"{counts['word']} Word files by their text" if counts["word"] else "",
            f"{counts['figure']} figure files by their source data" if counts["figure"] else "",
            f"{counts['other']} other files by their bytes" if counts["other"] else "",
        ]
        parts = [part for part in parts if part]
        every = "every file it could compare" if self.unread else "every file"
        return (
            "rebuild",
            "ok",
            f"Rebuilt from {copied} alone in a scratch folder, and {every} matches: "
            f"{join(parts) or 'no files to compare'}.",
        )


def _is_text(path: Path) -> bool:
    try:
        path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _excerpt(first: str, second: str) -> tuple[str, str]:
    """Two differing values, each cut to the stretch around their first difference when long."""
    if len(first) <= _WIDTH and len(second) <= _WIDTH:
        return first, second
    begin = max(0, len(os.path.commonprefix([first, second])) - 20)

    def cut(text: str) -> str:
        return (
            ("…" if begin else "")
            + text[begin : begin + _WIDTH]
            + ("…" if begin + _WIDTH < len(text) else "")
        )

    return cut(first), cut(second)


def _plural(count: int, word: str) -> str:
    return f"{count:,} {word}" + ("" if count == 1 else "s")


def compare_table(
    disk: Path, rebuilt: Path, *, rtol: float = RTOL, atol: float = ATOL, separator: str = ","
) -> tuple[list[str], int]:
    """Up to :data:`SHOWN` cell differences between two CSV files, and how many there are in all.

    Columns and rows must match. Cells that read as numbers on both sides match when
    ``numpy.isclose(rebuilt, disk, rtol, atol)`` holds; other cells must be equal as text. Rows are
    named by their line in the file (the header is line 1) and by the text cells that identify
    them, such as "Viability (%), Control".
    """
    try:
        old = pd.read_csv(disk, sep=separator, dtype=str, keep_default_na=False)
        new = pd.read_csv(rebuilt, sep=separator, dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        differences, total, _ = compare_text(disk, rebuilt)
        return differences, total
    if list(old.columns) != list(new.columns):
        return [
            f"columns on disk: {', '.join(map(str, old.columns))}",
            f"columns rebuilt: {', '.join(map(str, new.columns))}",
        ], 1
    examples, total = [], 0
    if len(old) != len(new):
        examples.append(f"{_plural(len(old), 'row')} on disk, {len(new):,} rebuilt")
        total += 1
    rows = min(len(old), len(new))
    old, new = old.iloc[:rows].reset_index(drop=True), new.iloc[:rows].reset_index(drop=True)
    unequal = old.ne(new)
    mismatch = unequal.copy()
    for column in old.columns[unequal.any().to_numpy()]:
        first = pd.to_numeric(old[column], errors="coerce").to_numpy(dtype=float)
        second = pd.to_numeric(new[column], errors="coerce").to_numpy(dtype=float)
        numeric = ~np.isnan(first) & ~np.isnan(second)
        close = np.zeros(len(first), dtype=bool)
        close[numeric] = np.isclose(second[numeric], first[numeric], rtol=rtol, atol=atol)
        mismatch[column] = unequal[column].to_numpy() & ~close
    positions = np.argwhere(mismatch.to_numpy())
    total += len(positions)
    labels = _row_labels(old)
    for row, column in positions[: SHOWN - len(examples)]:
        label = f" ({labels[row]})" if labels and labels[row] else ""
        before, after = _excerpt(old.iat[row, column], new.iat[row, column])
        examples.append(
            f"line {row + 2}{label}, column {old.columns[column]}: {before or '(empty)'} on disk, "
            f"{after or '(empty)'} rebuilt"
        )
    return examples, total


def _row_labels(frame: pd.DataFrame) -> list[str]:
    """Up to three leading text columns that tell rows apart, joined, for naming a row in a message."""
    chosen = []
    for column in frame.columns[:3]:
        values = frame[column]
        present = values[values != ""]
        if len(present) and pd.to_numeric(present, errors="coerce").notna().all():
            break
        chosen.append(column)
        if not frame[chosen].duplicated().any():
            break
    labels = [", ".join(value for value in row if value) for row in frame[chosen].itertuples(index=False)]
    return [label if len(label) <= 50 else label[:49] + "…" for label in labels]


def compare_text(disk: Path, rebuilt: Path) -> tuple[list[str], int, str]:
    """Up to :data:`SHOWN` line differences between two text files, their number, and any date ignored."""
    old = disk.read_text(encoding="utf-8", errors="replace").splitlines()
    new = rebuilt.read_text(encoding="utf-8", errors="replace").splitlines()
    ignored = ""
    rule = VOLATILE.get(disk.name)
    if rule:
        label, pattern = rule
        masked_old = [pattern.sub("…", line) for line in old]
        masked_new = [pattern.sub("…", line) for line in new]
        if old != new and masked_old == masked_new:
            ignored = label
        old, new = masked_old, masked_new
    differences = []
    matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        for offset in range(max(i2 - i1, j2 - j1)):
            first = old[i1 + offset] if i1 + offset < i2 else None
            second = new[j1 + offset] if j1 + offset < j2 else None
            if first is not None and second is not None:
                before, after = _excerpt(first, second)
                differences.append(f'line {i1 + offset + 1}: "{before}" on disk, "{after}" rebuilt')
            elif first is not None:
                differences.append(f'line {i1 + offset + 1}: "{_excerpt(first, "")[0]}" on disk only')
            else:
                differences.append(
                    f'line {j1 + offset + 1} of the rebuild: "{_excerpt(second, "")[0]}" only there'
                )
    return differences[:SHOWN], len(differences), ignored


def _word_text(path: Path) -> dict[str, str]:
    from docx import Document  # noqa: PLC0415

    document = Document(str(path))
    found = {f"paragraph {number}": paragraph.text for number, paragraph in enumerate(document.paragraphs, 1)}
    for t, table in enumerate(document.tables, 1):
        for r, row in enumerate(table.rows, 1):
            for c, cell in enumerate(row.cells, 1):
                found[f"table {t}, row {r}, column {c}"] = cell.text
    return found


def compare_word(disk: Path, rebuilt: Path) -> tuple[list[str], int]:
    """Up to :data:`SHOWN` differences in the text of two Word files, and how many there are."""
    try:
        old, new = _word_text(disk), _word_text(rebuilt)
    except Exception as error:  # noqa: BLE001
        return [f"could not be read as a Word file: {error}"], 1
    differences = []
    for place in dict.fromkeys([*new, *old]):
        if old.get(place) != new.get(place):
            before, after = _excerpt(old.get(place, "(missing)"), new.get(place, "(missing)"))
            differences.append(f'{place}: "{before}" on disk, "{after}" rebuilt')
    return differences[:SHOWN], len(differences)


def _summary(report: Report, record: dict) -> str:
    if report.ok:
        files = sum(len(entry.get("files", {})) for entry in record.get("assets", {}).values())
        return f"Everything matches: {len(record.get('assets', {}))} assets and their {files} files."
    counts = [
        f"{heading} ({len(report.failures(key))})"
        for key, heading in SECTIONS.items()
        if report.failures(key)
    ]
    return "Does not match: see " + join(counts) + " above."
