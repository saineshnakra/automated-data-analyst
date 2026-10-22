"""Where each asset came from, and what changed since the last build.

Every build writes ``provenance.json``: for each asset, its files and their
sha256, the input file's checksum, the rows it used and the exclusion steps
behind them, the parts of ``study.yml`` it depends on (hashed), the function
that made it, the calculation in words, references, and the package versions.
The next build compares itself with that record and writes ``CHANGES.md``,
naming each asset that changed and why.

Neither file holds a date, a clock time or anything else that changes between
two builds of the same study, so an unchanged rebuild leaves both as they were
and ``git diff`` stays quiet.
"""

from __future__ import annotations

import difflib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from .study import Study, sha256_of

MANIFEST = "provenance.json"
CHANGES = "CHANGES.md"
FORMAT = 1

#: Distributions whose versions can change a number or a figure.
PACKAGES = (
    "automated-data-analyst",
    "pandas",
    "numpy",
    "scipy",
    "statsmodels",
    "matplotlib",
    "python-docx",
    "PyYAML",
    "ruptures",
    "openpyxl",
    "python-calamine",
)
#: Files whose bytes say whether an asset changed between builds. Images and Word files are left
#: out: their bytes can change with fonts or the zip library while what they show stays the same.
TEXT_SUFFIXES = {".csv", ".md", ".tex", ".txt", ".json", ".cff", ".yml", ".yaml", ".sha256", ".py", ".sh"}


@dataclass
class Asset:
    id: str
    kind: str
    title: str
    files: list[Path] = field(default_factory=list)
    #: Parts of study.yml the asset is built from, such as "processing" or "analyses.primary".
    depends: tuple[str, ...] = ()
    calculation: str = ""
    references: tuple[str, ...] = ()
    code: str = ""
    #: Rows behind the asset: {"used": n, "excluded": n, "steps": [numbers]}.
    rows: dict[str, Any] = field(default_factory=dict)
    #: Headline numbers, so the change report can show old → new.
    numbers: dict[str, Any] = field(default_factory=dict)
    #: "Table 2" or "Figure 1" once numbered for the paper.
    number: str = ""
    caption: str = ""
    #: Whether the files list individual rows of the data.
    row_level: bool = False
    #: "main", "supplement" or "record": where the asset belongs in a paper.
    section: str = "main"


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def platform_name() -> str:
    """The operating system, processor and C library: what can change the last digits of a float.

    The kernel release is left out, since it changes with routine updates and never changes a result.
    """
    name = f"{platform.system()} {platform.machine()}"
    libc, version = platform.libc_ver()
    if libc:
        return f"{name}, {libc} {version}"
    mac = platform.mac_ver()[0]
    return f"{name}, macOS {mac}" if mac else name


def git_state(folder: Path) -> dict[str, Any]:
    """The commit and whether the working tree has uncommitted changes, when ``folder`` is in git.

    It is not written to ``provenance.json``: committing the assets would change the commit and so
    the record, and the next build would never match the last.
    """
    try:
        commit = subprocess.run(
            ["git", "-C", str(folder), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(folder), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {}
    return {"commit": commit, "uncommitted_changes": bool(dirty)}


def manifest(
    study: Study, out: Path, assets: list[Asset], processed, custom: dict[str, str]
) -> dict[str, Any]:
    from .. import __version__  # noqa: PLC0415

    def relative(path: Path) -> str:
        return path.resolve().relative_to(out).as_posix()

    records = []
    for record in processed.records:
        records.append(
            {
                "step": record.number,
                "kind": record.kind,
                "description": record.description,
                "reason": record.reason,
                "rows_before": record.rows_before,
                "rows_after": record.rows_after,
                "values_changed": record.values_changed,
            }
        )
    entries = {}
    for asset in assets:
        entries[asset.id] = {
            "kind": asset.kind,
            "title": asset.title,
            "number": asset.number,
            "section": asset.section,
            "caption": asset.caption,
            "files": {relative(path): sha256_of(path) for path in asset.files if path.is_file()},
            "depends_on": list(asset.depends),
            "config_sha256": study.section_hash(*asset.depends),
            "rows": asset.rows,
            "calculation": asset.calculation,
            "references": list(asset.references),
            "code": asset.code,
            "numbers": _plain(asset.numbers),
            "row_level": asset.row_level,
        }
    sections = sorted({key for asset in assets for key in asset.depends})
    return {
        "format": FORMAT,
        "ada": __version__,
        "python": platform.python_version(),
        "platform": platform_name(),
        "packages": package_versions(),
        "study": {"file": study.path.name, "sha256": sha256_of(study.path)},
        # Each part of study.yml on its own, so a later build or `ada verify` can name the parts that changed.
        "study_sections": {key: study.section_hash(key) for key in sections},
        "inputs": [
            {
                "path": study.data_path,
                "sheet": study.data_sheet,
                "sha256": processed.sha256,
                "rows": len(processed.original),
                "columns": len(processed.original.columns) - 1,
            }
        ],
        "custom": [{"script": script, "sha256": digest} for script, digest in custom.items()],
        "processing": records,
        "rows_used": len(processed.data),
        "assets": entries,
    }


def _plain(value):
    """Numbers as plain floats and ints so the record reads the same on every platform."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return float(f"{value:.12g}")
    return value


def read_manifest(out: Path) -> dict[str, Any] | None:
    path = out / MANIFEST
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_manifest(out: Path, record: dict[str, Any]) -> Path:
    """Write the record with its keys in the order built, which is the same at every build."""
    path = out / MANIFEST
    text = json.dumps(record, indent=1, sort_keys=False, ensure_ascii=False, default=str)
    path.write_text(text + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class Change:
    asset: str
    status: str  # new, removed, changed
    #: Why it changed (the inputs), then how (rows used, headline numbers, files), in plain words.
    reasons: tuple[str, ...]
    #: "Table 1" or "Figure 2", when the asset is numbered for the paper.
    number: str = ""


def compare(before: dict[str, Any] | None, after: dict[str, Any]) -> list[Change]:
    """Each asset that is new, removed or changed between two records, with why and how it changed."""
    if before is None:
        return []
    old_assets, new_assets = before.get("assets", {}), after.get("assets", {})
    general = general_causes(before, after, short=True)
    changes = []
    for ident, entry in new_assets.items():
        old = old_assets.get(ident)
        if old is None:
            changes.append(Change(ident, "new", (), entry.get("number", "")))
            continue
        effects = _effects(old, entry)
        if not effects:
            continue
        causes = asset_causes(before, after, ident) + general
        if not causes:
            causes = ["its files differ, though nothing it depends on changed"]
        changes.append(Change(ident, "changed", (*causes, *effects), entry.get("number", "")))
    for ident, old in old_assets.items():
        if ident not in new_assets:
            changes.append(Change(ident, "removed", (), old.get("number", "")))
    return changes


def changed_sections(before: dict[str, Any], after_sections: dict[str, str], keys) -> list[str] | None:
    """The parts of study.yml among ``keys`` whose hash differs from ``before``'s, or None if unknown."""
    then = before.get("study_sections")
    if then is None:
        return None
    return [key for key in keys if then.get(key) != after_sections.get(key)]


def sections_sentence(changed: list[str] | None, depends) -> str:
    if changed:
        return f"study.yml changed in {join(changed)}"
    return f"study.yml changed in one or more of {join(list(depends))}"


def asset_causes(before: dict[str, Any], after: dict[str, Any], ident: str) -> list[str]:
    """What changed among the inputs this one asset depends on: parts of study.yml, or its script."""
    old, new = before["assets"][ident], after["assets"][ident]
    causes = []
    if old.get("config_sha256") != new.get("config_sha256"):
        depends = new.get("depends_on", [])
        causes.append(
            sections_sentence(changed_sections(before, after.get("study_sections", {}), depends), depends)
        )
    old_scripts = {item["script"]: item["sha256"] for item in before.get("custom", [])}
    for item in after.get("custom", []):
        script = item["script"]
        if (
            new.get("code", "").startswith(f"{script}:")
            and old_scripts.get(script, item["sha256"]) != item["sha256"]
        ):
            causes.append(f"{script} changed")
    return causes


def general_causes(before: dict[str, Any], after: dict[str, Any], short: bool = False) -> list[str]:
    """Changes that can touch every asset: the data file, ADA, Python and the packages.

    ``short`` leaves out checksums and package names, for repeating on each asset's line.
    """
    causes = []
    old_input, new_input = (before.get("inputs") or [{}])[0], (after.get("inputs") or [{}])[0]
    if old_input.get("path") != new_input.get("path"):
        moved = f"the data file changed from {old_input.get('path')} to {new_input.get('path')}"
        same = old_input.get("sha256") == new_input.get("sha256")
        causes.append(moved + (" (same contents)" if same and not short else ""))
    elif old_input.get("sha256") != new_input.get("sha256"):
        causes.append(
            "the input data changed"
            + (
                ""
                if short
                else f" ({new_input.get('path')} sha256 {_digest(old_input.get('sha256'))} → "
                f"{_digest(new_input.get('sha256'))})"
            )
        )
    if before.get("ada") != after.get("ada"):
        causes.append(f"ADA {before.get('ada')} → {after.get('ada')}")
    if before.get("python") != after.get("python"):
        causes.append(f"Python {before.get('python')} → {after.get('python')}")
    old_packages = before.get("packages", {})
    changed_packages = [
        f"{name} {old_packages.get(name)} → {version}"
        for name, version in after.get("packages", {}).items()
        if old_packages.get(name) not in (None, version)
    ]
    if changed_packages:
        causes.append("package versions changed" + ("" if short else ": " + ", ".join(changed_packages)))
    return causes


def _effects(old: dict, new: dict) -> list[str]:
    """How an asset's output changed: rows used, headline numbers and files, or nothing."""
    effects = []
    if old.get("rows") != new.get("rows"):
        effects.append(_rows_change(old.get("rows") or {}, new.get("rows") or {}))
    effects.extend(_number_changes(old.get("numbers") or {}, new.get("numbers") or {}))
    old_files, new_files = old.get("files", {}), new.get("files", {})
    changed = [
        path
        for path, digest in new_files.items()
        if Path(path).suffix in TEXT_SUFFIXES and old_files.get(path) != digest
    ]
    gone = [path for path in old_files if path not in new_files]
    if changed:
        effects.append("files changed: " + ", ".join(changed))
    if gone:
        effects.append("files no longer written: " + ", ".join(gone))
    kept: list[str] = []
    for effect in effects:
        if effect and not any(effect in earlier for earlier in kept):
            kept.append(effect)
    return kept


def _rows_change(old: dict, new: dict) -> str:
    """The rows used before and after, such as "rows used 48 → 46 (excluded 0 → 2, by step 3)"."""
    detail = []
    if old.get("excluded") != new.get("excluded"):
        detail.append(f"excluded {_count(old.get('excluded'))} → {_count(new.get('excluded'))}")
    if old.get("exclusion_steps") != new.get("exclusion_steps"):
        steps = list(new.get("exclusion_steps") or [])
        if steps:
            detail.append(
                ("by step " if len(steps) == 1 else "by steps ") + join([str(step) for step in steps])
            )
        else:
            detail.append("no exclusion steps")
    parts = []
    if old.get("used") != new.get("used"):
        parts.append(
            f"rows used {_count(old.get('used'))} → {_count(new.get('used'))}"
            + (f" ({', '.join(detail)})" if detail else "")
        )
    elif detail:
        parts.append("rows " + ", ".join(detail))
    for key in [key for key in new if key not in ("used", "excluded", "exclusion_steps")] + [
        key for key in old if key not in new and key not in ("used", "excluded", "exclusion_steps")
    ]:
        if old.get(key) != new.get(key):
            parts.append(f"{_words(key)} {_value(old.get(key))} → {_value(new.get(key))}")
    return "; ".join(parts)


def _number_changes(old: dict, new: dict) -> list[str]:
    """Each headline number that changed, as "name old → new", shown to as many digits as tell them apart."""
    parts = []
    for key, value in new.items():
        if key in old and old[key] != value:
            first, second = _pair(old[key], value)
            parts.append(f"{_words(key)} {first} → {second}")
        elif key not in old:
            parts.append(f"{_words(key)} (new) {_value(value)}")
    parts += [
        f"{_words(key)} no longer reported (was {_value(value)})"
        for key, value in old.items()
        if key not in new
    ]
    return parts


def _pair(old, new) -> tuple[str, str]:
    if _is_number(old) and _is_number(new) and (isinstance(old, float) or isinstance(new, float)):
        for digits in (4, 6, 8, 10, 12, 17):
            first, second = f"{old:.{digits}g}", f"{new:.{digits}g}"
            if first != second:
                return first, second
    return _value(old), _value(new)


def _is_number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _value(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    if isinstance(value, list):
        return ", ".join(map(str, value)) or "none"
    return str(value)


def _count(value) -> str:
    return f"{value:,}" if _is_number(value) else _value(value)


def _words(key: str) -> str:
    return str(key).replace("_", " ")


def _digest(value) -> str:
    return f"{value[:12]}…" if value else "none"


def join(items: list[str]) -> str:
    """Items joined as in a sentence: "a", "a and b", "a, b and c"."""
    items = [str(item) for item in items]
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def name(ident: str, entry: dict) -> str:
    """An asset as a reader knows it: "descriptives (Table 1)", or just its id when it has no number."""
    number = entry.get("number", "") if entry else ""
    return f"{ident} ({number})" if number else ident


def step_changes(old_steps: list[dict], new_steps: list[dict]) -> list[str]:
    """The processing steps that are new, removed or changed, and those that now remove other rows."""

    def keys(steps: list[dict]) -> list[tuple]:
        return [(step.get("kind"), step.get("description")) for step in steps]

    matcher = difflib.SequenceMatcher(a=keys(old_steps), b=keys(new_steps), autojunk=False)
    lines = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        old_block, new_block = old_steps[i1:i2], new_steps[j1:j2]
        if tag == "equal":
            for old, new in zip(old_block, new_block, strict=True):
                if _step_effect(old) != _step_effect(new):
                    lines.append(
                        f"Step {new['step']} ({new['description']}) now {_step_effect(new)}, "
                        f"where it {_step_effect(old, past=True)}."
                    )
        elif tag == "replace" and len(old_block) == len(new_block):
            for old, new in zip(old_block, new_block, strict=True):
                lines.append(
                    f"Step {new['step']} changed from “{old['description']}” to “{new['description']}”: "
                    f"it now {_step_effect(new)}, where it {_step_effect(old, past=True)}."
                )
        else:
            lines += [f"Step {old['step']} was removed: {old['description']}." for old in old_block]
            lines += [
                f"Step {new['step']} is new: {new['description']}. "
                f"Reason: {_sentence(new.get('reason') or 'none given')} It {_step_effect(new)}."
                for new in new_block
            ]
    return lines


def _step_effect(step: dict, past: bool = False) -> str:
    removed = step.get("rows_before", 0) - step.get("rows_after", 0)
    if step.get("kind") == "exclude":
        verb = "removed" if past else "removes"
        return f"{verb} {removed:,} row" + ("" if removed == 1 else "s")
    if step.get("kind") == "read":
        verb = "read" if past else "reads"
        return f"{verb} {step.get('rows_after', 0):,} rows"
    changed = step.get("values_changed", 0)
    verb = "changed" if past else "changes"
    return f"{verb} {changed:,} value" + ("" if changed == 1 else "s")


def _what_changed(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """The changes to the inputs since the previous build, one sentence each."""
    lines = []
    after_sections = after.get("study_sections", {})
    sections = changed_sections(before, after_sections, sorted(after_sections))
    if sections:
        lines.append(f"study.yml changed in {join(sections)}.")
    elif before.get("study", {}).get("sha256") != after.get("study", {}).get("sha256"):
        lines.append(
            "study.yml was edited, but none of the parts the assets depend on changed."
            if sections is not None
            else "study.yml changed."
        )
    lines += [_sentence(cause) for cause in general_causes(before, after)]
    old_scripts = {item["script"]: item["sha256"] for item in before.get("custom", [])}
    new_scripts = {item["script"]: item["sha256"] for item in after.get("custom", [])}
    for script, digest in new_scripts.items():
        if script not in old_scripts:
            lines.append(f"{script} is a new custom script.")
        elif old_scripts[script] != digest:
            lines.append(f"{script} changed (sha256 {_digest(old_scripts[script])} → {_digest(digest)}).")
    lines += [f"{script} is no longer run." for script in old_scripts if script not in new_scripts]
    lines += step_changes(before.get("processing", []), after.get("processing", []))
    if before.get("rows_used") != after.get("rows_used"):
        lines.append(f"Rows used: {_count(before.get('rows_used'))} → {_count(after.get('rows_used'))}.")
    return lines


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:] + ("" if text.endswith(".") else ".")


def changes_markdown(changes: list[Change], before: dict | None, after: dict) -> str:
    """This build's entry for ``CHANGES.md``, or "" when nothing changed since the previous build.

    It says what changed among the inputs, then each asset that changed with why (the parts of
    study.yml, the data, a script or a version), its rows used and headline numbers old → new and
    the files that differ, and names every asset that stayed the same.
    """
    source = after["inputs"][0]
    built_from = (
        f"ADA {after.get('ada')}, from {source['path']} (sha256 {_digest(source['sha256'])}) and "
        f"{after.get('study', {}).get('file', 'study.yml')} "
        f"(sha256 {_digest(after.get('study', {}).get('sha256'))})"
    )
    if before is None:
        return (
            f"First build: {len(after['assets'])} assets by {built_from}, {after['rows_used']:,} rows used.\n"
        )
    inputs = _what_changed(before, after)
    if not changes and not inputs:
        return ""
    lines = [f"Built by {built_from}.", ""]
    if inputs:
        lines += ["Since the previous build:", ""] + [f"- {line}" for line in inputs] + [""]
    named = {change.asset for change in changes}
    if changes:
        lines += ["Assets that changed:", ""]
        for change in changes:
            entry = after["assets"].get(change.asset) or before["assets"].get(change.asset, {})
            number = f" ({change.number})" if change.number else ""
            if change.status == "changed":
                lines.append(f"- **{change.asset}** changed{number}: " + "; ".join(change.reasons) + ".")
            elif change.status == "new":
                lines.append(f"- **{change.asset}** is new{number}: {entry.get('title', '')}.")
            else:
                lines.append(f"- **{change.asset}** was removed{number}: study.yml no longer makes it.")
        lines.append("")
    unchanged = [ident for ident in after["assets"] if ident not in named]
    if unchanged:
        lines.append(
            "Unchanged: " + ", ".join(name(ident, after["assets"][ident]) for ident in unchanged) + "."
        )
        same_despite: dict[str, list[str]] = {}
        for ident in unchanged:
            if ident in before.get("assets", {}):
                for cause in asset_causes(before, after, ident):
                    same_despite.setdefault(cause, []).append(name(ident, after["assets"][ident]))
        for cause, names in same_despite.items():
            lines.append(f"Although {cause}, {join(names)} came out the same.")
        lines.append("")
    return "\n".join(lines)


def write_changes(out: Path, text: str) -> Path:
    """Put this build's entry at the top of ``CHANGES.md``, keeping the earlier entries.

    Entries are numbered rather than dated, so the same builds give the same file, and an
    unchanged rebuild (an empty ``text``) leaves the file as it is.
    """
    path = out / CHANGES
    previous = path.read_text(encoding="utf-8") if path.is_file() else ""
    if not text:
        if previous:
            return path
        text = "Nothing changed since the previous build.\n"
    parts = previous.split("\n", 2)
    body = (parts[2] if len(parts) > 2 else "") if previous.startswith("# Changes") else previous
    count = sum(1 for line in body.splitlines() if line.startswith("## Build"))
    entry = f"## Build {count + 1}\n\n{text}"
    path.write_text(
        "# Changes\n\n" + entry + ("\n" + body.lstrip("\n") if body.strip() else ""), encoding="utf-8"
    )
    return path
