"""The reproduction package, the files for a data or code repository, and what may be shared.

``ada build`` ends by writing two folders. ``reproduce/`` holds what another
researcher needs to rebuild every asset: copies of study.yml and the custom
scripts, the exact package versions, the input checksums and plain steps.
``repository/`` holds drafts for a data or code deposit: a README,
CITATION.cff, .zenodo.json and data and code availability statements. They
say what study.yml says and leave a marked placeholder for the rest; ADA never
chooses a licence.

``ada preview`` lists every file in the output folder and whether the
``sharing`` settings let it out. ``ada share`` copies only those files, with
withheld columns removed, then checks the copy again and refuses to finish if
anything withheld is left.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path, PurePosixPath

import pandas as pd

from . import provenance
from .processing import SOURCE_ROW, read
from .provenance import Asset
from .study import Study, StudyError, load, sha256_of
from .synthetic import demo_study, synthesize

REPRODUCE = "reproduce"
REPOSITORY = "repository"
#: Where ``ada share`` puts the original data file when ``sharing.data`` is ``raw``.
DATA_FOLDER = "data"
#: Written by ``ada share``: what was left out and why. It also marks the folder as one ADA may replace.
SHARE_RECORD = "SHARING.md"
DEMO_DATA = "demo-data.csv"
DEMO_README = "demo-data.README.md"
DEMO_STUDY = "demo-study.yml"
#: The demo study's output folder, beside demo-study.yml.
DEMO_OUTPUT = "demo-assets"
ADA_REPOSITORY = "https://github.com/saineshnakra/automated-data-analyst"
ADA_PYPI = "https://pypi.org/project/automated-data-analyst/"
#: Every part of study.yml: the reproduction package copies the whole file.
WHOLE = (
    "study",
    "data",
    "design",
    "variables",
    "processing",
    "decisions",
    "analyses",
    "figures",
    "descriptives",
    "output",
    "sharing",
    "custom",
)
#: Files that hold rows of a table, as opposed to reports, documents and figures.
DATA_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".parquet"}
#: Files searched for values of withheld columns.
TEXT_SUFFIXES = {
    ".csv",
    ".tsv",
    ".md",
    ".tex",
    ".txt",
    ".json",
    ".yml",
    ".yaml",
    ".cff",
    ".sh",
    ".py",
    ".svg",
}
FIGURE_SUFFIXES = {".pdf", ".svg", ".png", ".tiff", ".tif", ".eps"}
#: Files the build writes outside any asset.
KNOWN = {
    "README.md": "index of the assets",
    "CHANGES.md": "what changed at each build",
    "provenance.json": "where each asset came from: checksums, row counts and calculations, no data values",
    "figures/captions.md": "figure captions",
}
PROCESSED_DATA = "processing/processed-data.csv"
EXCLUDED_ROWS = "processing/excluded-rows.csv"
#: Common words that a withheld column may hold without the value being private.
COMMON_VALUES = {
    "none",
    "None",
    "unknown",
    "Unknown",
    "missing",
    "Missing",
    "other",
    "Other",
    "true",
    "false",
}
WORD = re.compile(r"\w+")
#: Longest value, in words, that the search for withheld values compares whole.
LONGEST_VALUE = 8


def run(result) -> None:
    """Write ``reproduce/`` and ``repository/`` into ``result.out`` and record each as an asset."""
    study = result.study
    inputs = input_files(result)
    result.assets.append(
        Asset(
            id="reproduction-package",
            kind="record",
            title="Reproduction package",
            files=_reproduce(result, inputs),
            depends=WHOLE,
            code="ada.research.package.run",
            rows=result.rows(),
            numbers={"inputs": len(inputs)},
            calculation="Copies of study.yml and the custom scripts; installed versions from "
            "importlib.metadata; sha256 of each input (hashlib).",
            section="package",
        )
    )
    if study.sharing.data == "demo":
        result.assets.append(_demo(result))
    result.assets.append(
        Asset(
            id="repository-metadata",
            kind="record",
            title="Repository metadata and availability statements",
            files=_repository(result),
            depends=("study", "data", "sharing", "custom"),
            code="ada.research.package.run",
            rows=result.rows(),
            calculation="Drafts filled in from study.yml; anything it does not say is left as a "
            "placeholder in [square brackets].",
            references=(
                "Druskat, S., Spaaks, J. H., Chue Hong, N. et al. (2021). Citation File Format (version "
                "1.2.0). https://doi.org/10.5281/zenodo.5171937",
            ),
            section="package",
        )
    )


def _write(path: Path, text: str) -> Path:
    """Write UTF-8 with Unix line ends, so the file is the same on every platform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


# The reproduction package.


def input_files(result) -> list[tuple[str, str]]:
    """Each file the build reads, as (path relative to the study folder, sha256)."""
    study = result.study
    entries = [(study.data_path, result.processed.sha256), (study.path.name, sha256_of(study.path))]
    for entry in study.custom:
        digest = result.custom.get(entry.script) or sha256_of(study.root / entry.script)
        entries.append((entry.script, digest))
    return entries


def script_copies(study: Study) -> list[tuple[str, str]]:
    """(path in study.yml, path inside ``reproduce/``) for each custom script."""
    copies = []
    for entry in study.custom:
        target = PurePosixPath(entry.script.replace("\\", "/"))
        if target.is_absolute() or ".." in target.parts:
            target = PurePosixPath("custom") / target.name
        copies.append((entry.script, target.as_posix()))
    return copies


def _reproduce(result, inputs: list[tuple[str, str]]) -> list[Path]:
    study = result.study
    folder = result.folder(REPRODUCE)
    copied = folder / study.path.name
    shutil.copyfile(study.path, copied)
    files = [copied]
    for script, target in script_copies(study):
        destination = folder / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(study.root / script, destination)
        files.append(destination)
    files.append(_write(folder / "requirements.txt", requirements()))
    files.append(_write(folder / "checksums.sha256", checksums(inputs)))
    files.append(_write(folder / "INSTRUCTIONS.md", instructions(result, inputs)))
    script = _write(folder / "reproduce.sh", reproduce_script(study))
    script.chmod(0o755)
    files.append(script)
    return files


def _normal(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def pinned_versions() -> dict[str, str]:
    """Installed versions of ADA's dependencies and of every package that can change a number."""
    versions = provenance.package_versions()
    seen = {_normal(name) for name in versions}
    try:
        declared = metadata.requires("automated-data-analyst") or []
    except metadata.PackageNotFoundError:
        declared = []
    for requirement in declared:
        if "extra ==" in requirement:
            continue
        match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
        if not match or _normal(match.group()) in seen:
            continue
        try:
            versions[match.group()] = metadata.version(match.group())
        except metadata.PackageNotFoundError:
            continue
        seen.add(_normal(match.group()))
    versions = {
        name: version for name, version in versions.items() if _normal(name) != "automated-data-analyst"
    }
    return dict(sorted(versions.items(), key=lambda item: item[0].lower()))


def requirements() -> str:
    from .. import __version__  # noqa: PLC0415

    lines = [
        "# The exact versions that built these assets. Install them in a new environment with",
        "#   pip install -r requirements.txt",
        f"# Python {platform.python_version()}",
        f"automated-data-analyst=={__version__}",
    ]
    lines += [f"{name}=={version}" for name, version in pinned_versions().items()]
    return "\n".join(lines) + "\n"


def checksums(inputs: list[tuple[str, str]]) -> str:
    """The inputs in ``sha256sum`` format, so ``sha256sum -c`` can check them from the study folder."""
    lines = []
    for path, digest in inputs:
        if "\\" in path or "\n" in path:
            # GNU coreutils' escape for awkward names: a leading backslash, then \\ and \n in the name.
            lines.append("\\" + digest + "  " + path.replace("\\", "\\\\").replace("\n", "\\n"))
        else:
            lines.append(f"{digest}  {path}")
    return "\n".join(lines) + "\n"


def _back_to_study(study: Study) -> str:
    """The study folder relative to ``reproduce/``, for the configured output folder."""
    try:
        return Path(os.path.relpath(study.root, study.out_dir / REPRODUCE)).as_posix()
    except ValueError:
        return "."


def reproduce_script(study: Study) -> str:
    title = " ".join(study.title.split())
    back = _back_to_study(study)
    build = "ada build" + (f" --study '{study.path.name}'" if study.path.name != "study.yml" else "")
    verify = "ada verify" + (f" --study '{study.path.name}'" if study.path.name != "study.yml" else "")
    return "\n".join(
        [
            "#!/bin/sh",
            f"# Rebuild the assets of {title} and check them against the published ones.",
            "# Usage: sh reproduce.sh [STUDY_FOLDER]",
            f"# STUDY_FOLDER holds study.yml and the data; by default it is {back} from this script.",
            "set -eu",
            'here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)',
            f'cd "${{1:-$here/{back}}}"',
            'echo "Checking the inputs against checksums.sha256"',
            "if command -v sha256sum >/dev/null 2>&1; then",
            '    sha256sum -c "$here/checksums.sha256"',
            "else",
            '    shasum -a 256 -c "$here/checksums.sha256"',
            "fi",
            build,
            verify,
            "",
        ]
    )


def _names(names) -> str:
    names = [str(name) for name in names]
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _columns(names, kind: str = "") -> str:
    names = list(names)
    kind = f"{kind} " if kind else ""
    return (f"the {kind}column " if len(names) == 1 else f"the {kind}columns ") + _names(names)


def _contact() -> str:
    return "[name and email of the corresponding author, or of the data access committee]"


def _shared(study: Study) -> str:
    """What ``ada share`` lets out: ``sharing.data``, except that ``row_level: false`` keeps any data back."""
    mode = study.sharing.data
    return "none" if mode in ("processed", "raw") and not study.sharing.row_level else mode


def _data_step(result) -> list[str]:
    study = result.study
    mode = _shared(study)
    withheld_in_file = [name for name in study.sharing.withhold if name in result.processed.original]
    path = f"`{study.data_path}`"
    rep = f"{study.output.dir}/{REPRODUCE}"
    if mode == "raw":
        lines = [
            f"The original data file is included with the shared files, as {DATA_FOLDER}/"
            f"{Path(study.data_path).name}. Put it at {path} in the study folder.",
        ]
        if withheld_in_file:
            lines += [
                "",
                f"The shared copy leaves out {_columns(withheld_in_file)}, so it is not the file the "
                "published build read (compare the checksums below) and `ada build` cannot rebuild from it "
                f"as it is. The complete file is available from {_contact()} on request.",
            ]
        return lines
    if mode == "processed":
        dropped = [name for name in study.sharing.withhold if name in result.data]
        return [
            f"The processed data are shared as processing/processed-data.csv: the {len(result.data):,} rows "
            "every table and figure was computed from, after the steps in the processing record"
            + (f", without {_columns(dropped)}." if dropped else "."),
            "",
            f"`ada build` starts from the original file, {path}, which is not shared. Request it from "
            f"{_contact()}. Without it you can check each analysis against the processed data, but not run "
            "steps 3 and 4.",
        ]
    lines = [
        "The data are not shared publicly because [reason, such as the terms of the participants' consent]. "
        f"To request access, contact {_contact()}, saying [what a request must include]. Access is subject "
        "to [conditions, such as a data use agreement].",
        "",
        f"Once you have the file, put it at {path} in the study folder.",
    ]
    if mode == "demo":
        lines += [
            "",
            f"**Demo data.** To try every step without the real data, use {DEMO_DATA} in this folder. It "
            "is SYNTHETIC: it has the columns, types and levels of the processed data, but every value was "
            "drawn at random column by column, so it shows how the build runs and none of its numbers are "
            f"the study's. {DEMO_README} says how it was made. Build it with",
            "",
            f"    ada build --study {rep}/{DEMO_STUDY}",
            "",
            f"(`reproduce/{DEMO_STUDY}` in the shared files), which writes {DEMO_OUTPUT}/ beside it.",
        ]
    return lines


def instructions(result, inputs: list[tuple[str, str]]) -> str:
    """Plain steps to rebuild and check the assets, for a reader who has never used ADA."""
    from .. import __version__  # noqa: PLC0415

    study = result.study
    out = study.output.dir
    rep = f"{out}/{REPRODUCE}"
    option = f" --study {study.path.name}" if study.path.name != "study.yml" else ""
    scripts = [target for _, target in script_copies(study)]
    holds = [f"{study.path.name}, which records every decision behind the assets"]
    if scripts:
        holds.append("the custom script" + ("s " if len(scripts) > 1 else " ") + _names(scripts))
    holds += [
        "the exact package versions (requirements.txt)",
        "the checksums of the inputs (checksums.sha256)",
    ]
    made = [
        asset
        for asset in result.assets
        if asset.kind in ("table", "figure") and asset.section in ("main", "supplement")
    ]
    exclusions = [record for record in result.processed.records if record.kind == "exclude"]
    lines = [
        f"# Reproducing the assets of {study.title}",
        "",
        f"These steps rebuild every table, figure and record in {out}/ from the data, and check that each "
        "number matches the published build.",
        "",
        "This folder holds copies of everything the build reads except the data: " + _names(holds) + ". "
        f"`sh {rep}/reproduce.sh` checks the inputs and runs steps 3 and 4 in one go.",
        "",
        f"Work in the study folder, the one holding {study.path.name} and the data. If you have only the "
        f"shared files, make one: copy {study.path.name}"
        + (" and the scripts" if scripts else "")
        + " from this folder into a new folder, put the data in it as step 1 says, and copy the shared "
        f"files into {out}/ inside it, so the build can compare itself with the published record.",
        "",
        "## 1. Get the data",
        "",
        *_data_step(result),
        "",
        "## 2. Install the exact versions",
        "",
        f"These assets were built with Python {platform.python_version()} and ADA {__version__}. In a new "
        "virtual environment:",
        "",
        "    python -m venv .venv",
        "    . .venv/bin/activate          # on Windows: .venv\\Scripts\\activate",
        f"    pip install -r {rep}/requirements.txt",
        "",
        "## 3. Build",
        "",
        f"    ada build{option}",
        "",
        (
            f"ADA checks the data file against `data.sha256` in {study.path.name} and stops if it differs. "
            if study.data_sha256
            else f"{study.path.name} records no `data.sha256`, so check the data file against the checksums "
            "below first. "
        )
        + f"The build then remakes every asset in {out}/ and compares them with the previous build's "
        "record.",
        "",
        "## 4. Verify",
        "",
        f"    ada verify{option}",
        "",
        "This builds again in a scratch folder and compares every table and source-data file, number by "
        f"number, with {out}/.",
        "",
        "## 5. What to expect",
        "",
        f"- The build reads {len(result.processed.original):,} rows from {study.data_path} and uses "
        f"{len(result.data):,}"
        + (
            f", after {len(exclusions)} exclusion step{'s' if len(exclusions) != 1 else ''} "
            "(processing/processing-record.md)."
            if exclusions
            else "."
        ),
    ]
    if made:
        lines.append("- It makes " + _names(f"{asset.number} ({asset.title})" for asset in made) + ".")
    lines += [
        f"- {out}/CHANGES.md gains an entry for your build. When the rebuild matches the published one it "
        "reads “Nothing changed since the previous build.”; otherwise it names each asset that changed, "
        "and why.",
        "- `ada verify` reports that every asset matches.",
        "",
        "Before building, confirm that the inputs are the ones the published build used "
        "(reproduce.sh does this with `sha256sum -c checksums.sha256`):",
        "",
        "| File | sha256 |",
        "|---|---|",
        *[f"| `{path}` | `{digest}` |" for path, digest in inputs],
        "",
    ]
    return "\n".join(lines)


def _demo(result) -> Asset:
    study = result.study
    folder = result.folder(REPRODUCE)
    synthetic = synthesize(study, result.data)
    data_path = _write(folder / DEMO_DATA, synthetic.data.to_csv(index=False, lineterminator="\n"))
    digest = sha256_of(data_path)
    text, changes = demo_study(study, synthetic, DEMO_DATA, digest, DEMO_OUTPUT)
    for script, target in script_copies(study):
        if script != target:
            text = text.replace(script, target)
    study_path = _write(folder / DEMO_STUDY, text)
    readme = _write(folder / DEMO_README, demo_readme(study, synthetic, digest, changes))
    return Asset(
        id="demo-data",
        kind="record",
        title="Synthetic demo data",
        files=[data_path, readme, study_path],
        depends=("data", "variables", "design", "processing", "sharing"),
        code="ada.research.synthetic.synthesize",
        rows=result.rows(),
        numbers={"rows": len(synthetic.data), "columns": len(synthetic.data.columns)},
        calculation="Each column drawn on its own: numbers by inverse-transform sampling from the observed "
        "distribution (numpy.quantile, linear interpolation), categories by a random permutation "
        f"(numpy.random.Generator.permutation), identifiers replaced; seed {synthetic.seed}.",
        references=("Devroye, L. (1986). Non-Uniform Random Variate Generation. Springer, chapter 2.",),
        section="package",
    )


def demo_readme(study: Study, synthetic, digest: str, changes: list[str]) -> str:
    frame = synthetic.data
    lines = [
        f"# {DEMO_DATA}: SYNTHETIC data",
        "",
        "**These data are synthetic. They are not the study's data, and no result should be drawn from "
        "them.**",
        "",
        f"ADA wrote {DEMO_DATA} from the processed data of {study.title} because study.yml says "
        "`sharing.data: demo`: the real data stay private, and this table lets anyone run the whole build.",
        "",
        f"- It has {len(frame):,} rows and the columns, types and levels of the processed data"
        + (
            f", except {_columns(synthetic.left_out)}, which "
            + ("is" if len(synthetic.left_out) == 1 else "are")
            + " withheld and left out."
            if synthetic.left_out
            else "."
        ),
    ]
    if synthetic.unit:
        lines.append(
            f"- The design is kept: each of the {synthetic.units:,} values of {synthetic.unit} has the same "
            "number of rows"
            + (f" and the same {study.time} values" if study.time in frame else "")
            + " as in the study, under a new code."
        )
    lines += [
        "- Every other column was drawn on its own, so relationships between columns, including any "
        "difference between groups, are not kept.",
        f"- The draws are seeded (seed {synthetic.seed}), so every build writes the same file: sha256 "
        f"{digest}.",
        "",
        "| Column | How it was made |",
        "|---|---|",
        *[f"| {name} | {how} |" for name, how in synthetic.methods.items()],
        "",
        f"To run the build on it, from the folder that holds {REPRODUCE}/:",
        "",
        f"    ada build --study {REPRODUCE}/{DEMO_STUDY}",
        "",
        f"{DEMO_STUDY} is study.yml with these changes: " + "; ".join(changes) + ".",
        "",
    ]
    return "\n".join(lines)


# Repository files.

PARTICLES = {
    "al",
    "bin",
    "da",
    "das",
    "de",
    "del",
    "della",
    "den",
    "der",
    "di",
    "do",
    "dos",
    "du",
    "el",
    "ibn",
    "la",
    "le",
    "te",
    "ten",
    "ter",
    "van",
    "von",
}
SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv"}


def split_name(name: str) -> dict[str, str]:
    """Given names, particle, family name and suffix of a person's name, where they can be told apart.

    "Ludwig van Beethoven" and "van Beethoven, Ludwig" both give given "Ludwig",
    particle "van" and family "Beethoven". A one-word name, such as a
    consortium's, comes back whole as ``name``.
    """
    text = " ".join(str(name).split())
    if "," in text:
        family, _, rest = text.partition(",")
        parts = [part.strip() for part in rest.split(",") if part.strip()]
        suffix = parts.pop() if len(parts) > 1 and parts[-1].lower() in SUFFIXES else ""
        words = family.split()
        start = 0
        while start < len(words) - 1 and words[start].lower() in PARTICLES:
            start += 1
        return {
            "given": " ".join(parts),
            "particle": " ".join(words[:start]),
            "family": " ".join(words[start:]),
            "suffix": suffix,
        }
    words = text.split()
    suffix = words.pop() if len(words) > 2 and words[-1].lower() in SUFFIXES else ""
    if len(words) < 2:
        return {"name": text}
    start = len(words) - 1
    while start > 1 and words[start - 1].lower() in PARTICLES:
        start -= 1
    return {
        "given": " ".join(words[:start]),
        "particle": " ".join(words[start : len(words) - 1]),
        "family": words[-1],
        "suffix": suffix,
    }


def orcid_id(value: str) -> str | None:
    """The 16-character ORCID iD in ``value``, or None when it is not one (ISO 7064 11,2 check digit)."""
    text = re.sub(r"^(https?://)?(www\.)?orcid\.org/", "", str(value).strip(), flags=re.IGNORECASE).upper()
    if not re.fullmatch(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]", text):
        return None
    digits = text.replace("-", "")
    total = 0
    for digit in digits[:-1]:
        total = (total + int(digit)) * 2
    check = (12 - total % 11) % 11
    return text if digits[-1] == ("X" if check == 10 else str(check)) else None


@dataclass(frozen=True)
class Person:
    name: str
    parts: dict = field(default_factory=dict)
    orcid: str = ""
    affiliation: str = ""
    #: Why the ORCID in study.yml was left out.
    problem: str = ""

    @classmethod
    def of(cls, author) -> Person:
        orcid, problem = "", ""
        if author.orcid:
            orcid = orcid_id(author.orcid) or ""
            if not orcid:
                problem = (
                    f"the ORCID given for {author.name}, {author.orcid}, is not a valid ORCID iD, so it is "
                    "left out"
                )
        return cls(author.name, split_name(author.name), orcid, author.affiliation, problem)

    @property
    def family_first(self) -> str:
        """The name as repositories list it: "van Beethoven, Ludwig"."""
        if "name" in self.parts:
            return self.parts["name"]
        family = " ".join(part for part in (self.parts["particle"], self.parts["family"]) if part)
        return f"{family}, {self.parts['given']}" + (
            f", {self.parts['suffix']}" if self.parts["suffix"] else ""
        )

    def cff(self) -> dict:
        if "name" in self.parts:
            entry = {"name": self.parts["name"]}
        else:
            entry = {"given-names": self.parts["given"]} if self.parts["given"] else {}
            if self.parts["particle"]:
                entry["name-particle"] = self.parts["particle"]
            entry["family-names"] = self.parts["family"]
            if self.parts["suffix"]:
                entry["name-suffix"] = self.parts["suffix"]
        if self.orcid:
            entry["orcid"] = f"https://orcid.org/{self.orcid}"
        if self.affiliation:
            entry["affiliation"] = self.affiliation
        return entry

    def zenodo(self) -> dict:
        entry = {"name": self.family_first}
        if self.affiliation:
            entry["affiliation"] = self.affiliation
        if self.orcid:
            entry["orcid"] = self.orcid
        return entry


def related(study: Study) -> list[dict]:
    """Identifiers of related work from ``study.related``, such as the DOI of the paper the assets support.

    Each entry is a DOI or URL, or ``{identifier, relation}`` with a DataCite
    relation such as ``isSupplementTo`` (the default) or ``isDocumentedBy``.
    """
    entries = (study.raw.get("study") or {}).get("related") or []
    if isinstance(entries, str | dict):
        entries = [entries]
    found = []
    for entry in entries:
        if isinstance(entry, dict):
            identifier = entry.get("identifier") or entry.get("doi") or entry.get("url")
            relation = str(entry.get("relation") or "isSupplementTo")
        else:
            identifier, relation = entry, "isSupplementTo"
        if not identifier:
            continue
        identifier = re.sub(
            r"^(https?://)?(dx\.)?doi\.org/", "", str(identifier).strip(), flags=re.IGNORECASE
        )
        found.append({"identifier": identifier, "relation": relation})
    return found


def _ada_reference() -> dict:
    from .. import __version__  # noqa: PLC0415

    return {
        "type": "software",
        "title": "ADA: Automated Data Analyst",
        "authors": [{"family-names": "Nakra", "given-names": "Sainesh"}],
        "version": __version__,
        "repository-code": ADA_REPOSITORY,
        "url": ADA_PYPI,
        "license": "MIT",
    }


def citation_cff(study: Study) -> str:
    """A CITATION.cff (format 1.2.0) for the deposit. The licence is left out when study.yml has none."""
    import yaml  # noqa: PLC0415

    people = [Person.of(author) for author in study.authors]
    notes = [person.problem for person in people if person.problem]
    authors = [person.cff() for person in people]
    if not authors:
        authors = [{"name": "[authors]"}]
        notes.append("study.authors is empty; the author above is a placeholder")
    record = {
        "cff-version": "1.2.0",
        "message": "If you use these data or this code, please cite them using these metadata.",
        "title": study.title,
        "type": "dataset",
        "authors": authors,
    }
    if study.description:
        record["abstract"] = study.description
    if study.keywords:
        record["keywords"] = list(study.keywords)
    if study.license:
        record["license"] = study.license
    else:
        notes.append("study.license is blank and ADA does not choose a licence; add `license:` once you have")
    record["references"] = [_ada_reference()]
    header = [
        "# CITATION.cff draft written by ADA from study.yml. Check it before publishing, and add",
        "# `doi:` and `date-released:` once the deposit has them.",
        *[f"# TODO: {note}." for note in notes],
    ]
    return "\n".join(header) + "\n" + yaml.safe_dump(record, sort_keys=False, allow_unicode=True, width=100)


def _summary(result) -> str:
    """One paragraph saying what the deposit holds, for its description."""
    study = result.study
    mode = _shared(study)
    held = {
        "processed": "The processed data are included.",
        "raw": "The original and the processed data are included.",
        "demo": "The data are not included; a synthetic table with the same columns lets the build run.",
        "none": "The data are not included; they are available on request.",
    }[mode]
    return (
        f"Tables, figures and records for {study.title}, built by ADA from {Path(study.data_path).name} "
        f"({len(result.data):,} rows used; one row is {study.observation}). {held} "
        "reproduce/INSTRUCTIONS.md explains how to rebuild every asset."
    )


def zenodo_json(result) -> str:
    """Deposit metadata in Zenodo's ``.zenodo.json`` form. The licence is left out when study.yml has none."""
    from .. import __version__  # noqa: PLC0415

    study = result.study
    creators = [Person.of(author).zenodo() for author in study.authors] or [{"name": "[Family name, Given]"}]
    record = {
        "title": study.title,
        "upload_type": "dataset",
        "description": study.description or _summary(result),
        "creators": creators,
    }
    if study.keywords:
        record["keywords"] = list(study.keywords)
    if study.license:
        record["license"] = study.license.lower()
    links = related(study)
    if links:
        record["related_identifiers"] = links
    record["notes"] = f"Built with ADA {__version__} ({ADA_REPOSITORY})."
    return json.dumps(record, indent=2, ensure_ascii=False) + "\n"


def _licence_words(value: str) -> str:
    return f"the {value} licence" if value else "the [licence] licence"


def data_availability(result) -> str:
    """A draft data availability statement in the form journals ask for, for ``sharing.data``."""
    study = result.study
    mode, withheld = study.sharing.data, list(study.sharing.withhold)
    where = "in [repository name] at [repository DOI]"
    request = (
        f"They are available from {_contact()} on reasonable request, subject to [conditions, such as a data "
        "use agreement]."
    )
    consent = "[reason, such as the terms of the participants' consent]"
    licence = _licence_words(study.license)
    if mode == "processed":
        statement = (
            "The processed data that support the findings of this study are openly available "
            f"{where}, under {licence}. They hold the {len(result.data):,} observations analysed (each "
            f"row is {study.observation}), after the exclusions described in the Methods, with a data "
            "dictionary and a record of every processing step. The original data file is available from "
            "[the corresponding author] on reasonable request."
        )
    elif mode == "raw":
        statement = (
            "The original data file and the processed data that support the findings of this study are "
            f"openly available {where}, under {licence}. The deposit also holds a data dictionary, a "
            "record of every processing step and the checksum of the original file, so every exclusion "
            "can be checked."
        )
    elif mode == "demo":
        statement = (
            "The data that support the findings of this study are not openly available because "
            f"{consent}. {request} A synthetic dataset with the same columns, types and levels as the "
            "processed data, its values drawn at random, is openly available with the analysis code "
            f"{where}, so the full analysis can be run; it does not reproduce the reported results."
        )
    else:
        statement = (
            "The data that support the findings of this study are not openly available because "
            f"{consent}. {request} The data dictionary, the processing record and the analysis code are "
            f"openly available {where}."
        )
    lines = [
        "# Data availability statement (draft)",
        "",
        f"Written by ADA from study.yml (`sharing.data: {mode}`"
        + (f", `withhold: [{', '.join(withheld)}]`" if withheld else "")
        + "). Replace each part in [square brackets], and check the wording against the journal's data "
        "policy.",
        "",
        statement,
        "",
    ]
    if withheld and mode in ("processed", "raw"):
        lines += [
            f"The shared files leave out {_columns(withheld)} because [reason, such as they could identify "
            "participants].",
            "",
        ]
    elif withheld and mode == "demo":
        lines += [f"The synthetic dataset leaves out {_columns(withheld)}.", ""]
    notes = []
    if not study.license and mode in ("processed", "raw", "demo"):
        notes.append(
            "`study.license` is blank, so the statement says [licence]. ADA does not choose a licence for "
            "you; add it to study.yml and build again."
        )
    if mode in ("processed", "raw") and not study.sharing.row_level:
        notes.append(
            "`sharing.row_level` is false, so `ada share` keeps the data files back even though "
            f"`sharing.data` is {mode}; change one setting or the other so the statement holds."
        )
    if mode == "demo":
        notes.append(
            f"The synthetic dataset is {REPRODUCE}/{DEMO_DATA}; {REPRODUCE}/{DEMO_README} says how it was "
            "made."
        )
    notes.append("`ada preview` lists what `ada share` copies for the deposit, and `ada share` writes it.")
    lines += ["## Notes for the authors", "", *[f"- {note}" for note in notes], ""]
    return "\n".join(lines)


def code_availability(result) -> str:
    """A draft code availability statement: where the code is, what ran it, and what it needs."""
    from .. import __version__  # noqa: PLC0415

    study = result.study
    mode = study.sharing.data
    scripts = [entry.script for entry in study.custom]
    running = {
        "processed": "Rebuilding from the start needs the original data file, which is available on "
        "request; the processed data in the same deposit let each analysis be checked.",
        "raw": "The original data are in the same deposit, so every asset can be rebuilt from them.",
        "demo": "The data are not openly available, but a synthetic dataset in the same deposit lets the "
        f"whole pipeline run (`ada build --study {REPRODUCE}/{DEMO_STUDY}`).",
        "none": "Running it needs the data, which are available on request (see the data availability "
        "statement).",
    }[mode]
    statement = (
        "The code that produced the results is openly available in [repository name] at [repository "
        f"DOI], under {_licence_words(study.code_license)}. All processing, analyses, tables and figures "
        f"were produced with ADA {__version__} (automated-data-analyst; {ADA_REPOSITORY}) on Python "
        f"{platform.python_version()}, with the package versions in {REPRODUCE}/requirements.txt. "
        f"{study.path.name} records every processing step, statistical decision and analysis"
        + (
            f", and the custom script{'s' if len(scripts) > 1 else ''} {_names(scripts)} add further assets"
            if scripts
            else ""
        )
        + ". Running `ada build` regenerates every table and figure, and `ada verify` checks that each "
        f"number matches. {running}"
    )
    lines = [
        "# Code availability statement (draft)",
        "",
        "Written by ADA from study.yml. Replace each part in [square brackets].",
        "",
        statement,
        "",
    ]
    if not study.code_license:
        lines += [
            "## Notes for the authors",
            "",
            "- `study.code_license` is blank, so the statement says [licence]. ADA does not choose a "
            "licence for you; add one, such as the licence of the repository holding the code, to "
            "study.yml and build again.",
            "",
        ]
    return "\n".join(lines)


FOLDERS = {
    "tables": "tables as CSV with full-precision numbers, and as Word and LaTeX",
    "figures": "figures, the data behind each (.source.csv) and their captions",
    "methods": "the methods draft",
    "supplement": "exploratory analyses, model output and diagnostics",
    "audit": "the data audit",
    "dictionary": "the data dictionary",
    "processing": "the processing record: every exclusion and change, with its reason",
    "reproduce": "study.yml, the exact package versions, the input checksums and the steps to rebuild "
    "everything",
    "repository": "this README, CITATION.cff, .zenodo.json and drafts of the availability statements",
}


def repository_readme(result) -> str:
    """The README for a data or code deposit: who, what, how to rebuild, the licence and how to cite."""
    from .. import __version__  # noqa: PLC0415

    study = result.study
    people = [Person.of(author) for author in study.authors]
    todo = [person.problem[0].upper() + person.problem[1:] + "." for person in people if person.problem]
    lines = [f"# {study.title}", ""]
    if people:
        for person in people:
            details = [f"ORCID [{person.orcid}](https://orcid.org/{person.orcid})"] if person.orcid else []
            details += [person.affiliation] if person.affiliation else []
            lines.append(f"- {person.name}" + (f" ({'; '.join(details)})" if details else ""))
    else:
        lines.append("- [authors]")
        todo.append("Add the authors under `study.authors` in study.yml.")
    lines.append("")
    if study.description:
        lines += [study.description, ""]
    else:
        lines += ["[A short description of the study.]", ""]
        todo.append("Add a short description under `study.description` in study.yml.")
    lines += [
        "> Draft written by ADA from study.yml for a data or code repository. Replace anything in [square "
        "brackets] and check every statement before publishing. CITATION.cff and .zenodo.json go at the top "
        "of the repository.",
        "",
        _summary(result),
        "",
        "## What the files are",
        "",
        "Paths are those of the folder `ada share` writes; SHARING.md there lists any file left out, "
        "and why.",
        "",
        "| Path | What it is |",
        "|---|---|",
        "| `README.md` | every asset, with its caption and files |",
        "| `provenance.json` | where each asset came from: input checksums, rows used, calculations, package "
        "versions |",
        "| `CHANGES.md` | what changed at each build, and why |",
    ]
    present = {path.relative_to(result.out).parts[0] for asset in result.assets for path in asset.files}
    present |= {REPRODUCE, REPOSITORY}
    for folder, words in FOLDERS.items():
        if folder not in present:
            continue
        if folder == "processing" and study.sharing.data in ("processed", "raw") and study.sharing.row_level:
            words += ", and the processed data"
        if folder == REPRODUCE and study.sharing.data == "demo":
            words += f", and {DEMO_DATA}, a SYNTHETIC table for trying the build"
        lines.append(f"| `{folder}/` | {words} |")
    if study.sharing.data == "raw":
        lines.append(f"| `{DATA_FOLDER}/` | the original data file, {Path(study.data_path).name} |")
    lines += [
        "",
        "## How to reproduce",
        "",
        f"Install the exact versions in {REPRODUCE}/requirements.txt, put the data at "
        f"`{study.data_path}` next to study.yml, then run `ada build` and `ada verify`. "
        f"{REPRODUCE}/INSTRUCTIONS.md has the steps, including how to get the data.",
        "",
        "## Licence",
        "",
    ]
    if study.license:
        lines += [f"The data and assets are shared under {study.license}.", ""]
    else:
        lines.append(
            "**TODO: choose a licence for the data and assets.** `study.license` in study.yml is blank, and "
            "ADA does not choose one for you. Without a licence, others have no clear right to reuse the "
            "files. Once you have chosen, add it to study.yml and build again; CITATION.cff and "
            ".zenodo.json then include it."
        )
        lines.append("")
        todo.append("Choose a licence for the data and assets (`study.license`).")
    if study.code_license:
        lines.append(f"The code (study.yml and any scripts) is shared under {study.code_license}.")
    else:
        lines.append("**TODO: choose a licence for the code** (`study.code_license`).")
        todo.append("Choose a licence for the code (`study.code_license`).")
    authors = "; ".join(person.family_first for person in people) or "[authors]"
    lines += [
        "",
        "## How to cite",
        "",
        f"{authors} ([year]). {study.title} [Data set]. [repository name]. [repository DOI]",
        "",
        "CITATION.cff holds the same details in machine-readable form. Please also cite the software "
        f"that built the assets: Nakra, S. ADA: Automated Data Analyst (version {__version__}). "
        f"{ADA_REPOSITORY}",
        "",
    ]
    if todo:
        lines += ["## To do before publishing", "", *[f"- {item}" for item in todo], ""]
    return "\n".join(lines)


def _repository(result) -> list[Path]:
    folder = result.folder(REPOSITORY)
    return [
        _write(folder / "README.md", repository_readme(result)),
        _write(folder / "CITATION.cff", citation_cff(result.study)),
        _write(folder / ".zenodo.json", zenodo_json(result)),
        _write(folder / "data-availability.md", data_availability(result)),
        _write(folder / "code-availability.md", code_availability(result)),
    ]


# What may be shared.


@dataclass(frozen=True)
class Decision:
    """Whether one file may be shared, and why."""

    path: str
    size: int
    #: "share", "trim" (share with the withheld columns removed) or "leave".
    action: str
    reason: str
    #: Withheld columns to remove before sharing.
    columns: tuple[str, ...] = ()
    #: Things to check, shown with a mark in the preview.
    flags: tuple[str, ...] = ()
    #: A withheld value found in the file. Printed on screen only, never written to the shared folder.
    example: str = ""


class _Rules:
    """The sharing settings of one study, applied file by file."""

    def __init__(self, study: Study, record: dict, original: pd.DataFrame | None):
        self.study = study
        self.sharing = study.sharing
        self.withheld = set(study.sharing.withhold)
        self.restricted = study.sharing.data in ("none", "demo")
        self.data_allowed = study.sharing.data in ("processed", "raw") and study.sharing.row_level
        names = [name for name, variable in study.variables.items() if variable.role == "identifier"]
        self.identifiers = [name for name in dict.fromkeys([*names, study.subject, study.cluster]) if name]
        self.owners = {
            path: entry for entry in record.get("assets", {}).values() for path in entry.get("files", {})
        }
        self.values = _withheld_values(study, original) if original is not None else {}
        #: For each first word of a withheld value, the lengths of the values that start with it.
        self.firsts: dict[str, list[int]] = {}
        for value in self.values:
            self.firsts.setdefault(value[0], []).append(len(value))

    def decide(self, folder: Path, rel: str, rescan: bool = False) -> Decision:
        path = folder / rel
        size = path.stat().st_size
        suffix = path.suffix.lower()
        entry = self.owners.get(rel)
        mode = self.sharing.data
        header = _header(path) if suffix in (".csv", ".tsv") else []
        if header is None:
            return Decision(
                rel, size, "leave", "could not be read as a table to check its columns", flags=("unreadable",)
            )
        kind = _data_kind(rel, header)
        if rel in KNOWN:
            reason = KNOWN[rel]
        elif rescan and rel == SHARE_RECORD:
            reason = "what was left out of this folder, and why"
        elif rel.startswith(f"{DATA_FOLDER}/") and rescan:
            reason = "the original data file"
        elif entry is None and not rescan:
            return Decision(
                rel,
                size,
                "leave",
                "not made by the last build (not in provenance.json); copy it by hand if it should be shared",
            )
        else:
            reason = _label(entry) if entry else "copied by ada share"
        if kind and not (self.data_allowed and (kind != "original data file" or mode == "raw")):
            if mode not in ("processed", "raw"):
                why = f"the data are not shared (sharing.data: {mode})"
            elif not self.sharing.row_level:
                why = "it lists individual rows (sharing.row_level: false)"
            else:
                why = "only the processed data are shared (sharing.data: processed)"
            return Decision(rel, size, "leave", f"{kind}; {why}", flags=(f"{kind} kept back",))
        if entry and entry.get("row_level") and not self.sharing.row_level and suffix in DATA_SUFFIXES:
            return Decision(
                rel,
                size,
                "leave",
                f"{reason}; lists individual rows (sharing.row_level: false)",
                flags=("individual rows kept back",),
            )
        columns = tuple(name for name in header if name in self.withheld)
        found = self._search(path, suffix, columns)
        if found:
            value, column = found
            return Decision(
                rel,
                size,
                "leave",
                f"contains a value of the withheld column {column}",
                flags=(f"value of withheld column {column}",),
                example=value,
            )
        flags = []
        if columns:
            flags.append(
                f"has {_columns(columns, 'withheld')}; "
                f"`ada share` removes {'it' if len(columns) == 1 else 'them'}"
            )
        if self.restricted and rel != f"{REPRODUCE}/{DEMO_DATA}":
            named = [name for name in header if name in self.identifiers and name not in self.withheld]
            if named:
                flags.append(
                    f"has {_columns(named, 'identifier')} while the data are not shared "
                    f"(sharing.data: {mode}); add it to sharing.withhold if it must stay private"
                )
            if "removed_ids" in header and _has_values(path, "removed_ids"):
                flags.append(
                    "lists the identifiers of removed rows (removed_ids) while the data are not shared"
                )
        if kind:
            reason = f"{kind}, shared because sharing.data is {mode}"
        elif rel == f"{REPRODUCE}/{DEMO_DATA}":
            reason = f"{reason}: synthetic values only"
        elif rel.startswith(f"{REPRODUCE}/"):
            reason = f"{reason}: for rebuilding the assets"
        elif rel.startswith(f"{REPOSITORY}/"):
            reason = f"{reason}: draft for the deposit"
        elif entry and entry.get("row_level") and suffix in DATA_SUFFIXES:
            reason = f"{reason}: lists individual rows, allowed by sharing.row_level"
        elif entry and entry.get("kind") == "figure" and suffix in FIGURE_SUFFIXES:
            reason = f"{reason}: figure"
        elif entry:
            reason = (
                f"{reason}: report" if entry.get("row_level") else f"{reason}: summary, no individual rows"
            )
        return Decision(
            rel, size, "trim" if columns else "share", reason, columns=columns, flags=tuple(flags)
        )

    def _search(self, path: Path, suffix: str, columns: tuple[str, ...]) -> tuple[str, str] | None:
        """The first withheld value found in the file's text, after removing the withheld columns."""
        if not self.values:
            return None
        if suffix in (".csv", ".tsv") and columns:
            text = _without(path, columns)
        elif suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
        elif suffix == ".docx":
            text = _docx_text(path)
        else:
            return None
        tokens = WORD.findall(text)
        for start, token in enumerate(tokens):
            for length in self.firsts.get(token, ()):
                value = tuple(tokens[start : start + length])
                if value in self.values:
                    return " ".join(value), self.values[value]
        return None


def _withheld_values(study: Study, original: pd.DataFrame) -> dict[tuple[str, ...], str]:
    """Distinctive text values of the withheld columns, as word sequences, with their column.

    Short values (under four characters), numbers, dates and common words such
    as "unknown" are skipped: they would match ordinary text.
    """
    others = set()
    for name in original.columns:
        if name not in study.sharing.withhold and name != SOURCE_ROW:
            others.update(original[name].dropna().astype(str).str.strip())
    found: dict[tuple[str, ...], str] = {}
    for name in study.sharing.withhold:
        if name not in original:
            continue
        for value in original[name].dropna().astype(str).str.strip().unique():
            words = tuple(WORD.findall(value))[:LONGEST_VALUE]
            if (
                len(value) < 4
                or not words
                or all(word.isdigit() for word in words)
                or value in COMMON_VALUES
                or value in others
                or _is_number(value)
            ):
                continue
            found.setdefault(words, name)
    return found


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _label(entry: dict) -> str:
    if entry.get("number"):
        return f"{entry['number']}, {entry['title']}"
    return entry.get("title") or "an asset"


def _data_kind(rel: str, header: list[str]) -> str:
    if rel == PROCESSED_DATA:
        return "processed data"
    if rel == EXCLUDED_ROWS:
        return "excluded rows of the original data"
    if rel.startswith(f"{DATA_FOLDER}/"):
        return "original data file"
    if SOURCE_ROW in header:
        return "rows of the data"
    return ""


def _header(path: Path) -> list[str] | None:
    try:
        frame = pd.read_csv(path, nrows=0, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    except pd.errors.EmptyDataError:
        return []
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError):
        return None
    return [str(name) for name in frame.columns]


def _table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)


def _without(path: Path, columns) -> str:
    """The CSV's text with ``columns`` removed, every other cell kept as written."""
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return _table(path).drop(columns=list(columns)).to_csv(index=False, sep=sep, lineterminator="\n")


def _has_values(path: Path, column: str) -> bool:
    try:
        return bool((_table(path)[column].str.strip() != "").any())
    except (KeyError, ValueError, pd.errors.ParserError):
        return False


def _docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.read("word/document.xml").decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, KeyError, OSError):
        return ""


def _size(count: int) -> str:
    if count < 1000:
        return f"{count} B"
    for unit in ("kB", "MB", "GB"):
        count /= 1000
        if count < 1000 or unit == "GB":
            return f"{count:.1f} {unit}"
    return f"{count:.1f} GB"


def _files(folder: Path) -> list[str]:
    return sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file())


@dataclass
class Review:
    """Every file in the output folder, with whether it may be shared."""

    study: Study
    folder: Path
    decisions: list[Decision]
    notes: list[str] = field(default_factory=list)
    rules: _Rules | None = field(default=None, repr=False)

    @property
    def flagged(self) -> list[Decision]:
        return [decision for decision in self.decisions if decision.flags]

    def text(self) -> str:
        sharing = self.study.sharing
        settings = f"sharing.data: {sharing.data}, row_level: {str(sharing.row_level).lower()}"
        settings += f", withhold: {', '.join(sharing.withhold)}" if sharing.withhold else ", withhold: none"
        labels = {"share": "share", "trim": "share*", "leave": "leave out"}
        lines = [
            f"What would be shared from {self.folder.name}/ for {self.study.title}",
            f"Settings in {self.study.path.name}: {settings}.",
            "",
        ]
        for decision in self.decisions:
            mark = "!" if decision.flags else " "
            label, size = labels[decision.action], _size(decision.size)
            lines.append(f"{mark} {label:<9} {size:>8}  {decision.path}: {decision.reason}")
        if any(decision.action == "trim" for decision in self.decisions):
            lines += ["", "share*: shared with the withheld columns removed."]
        if self.flagged:
            lines += ["", f"Flagged ({len(self.flagged)}):"]
            for decision in self.flagged:
                for flag in decision.flags:
                    example = f" (such as “{decision.example}”)" if decision.example else ""
                    lines.append(f"  ! {decision.path}: {flag}{example}")
        lines += [f"note: {note}" for note in self.notes]
        shared = [decision for decision in self.decisions if decision.action != "leave"]
        trimmed = sum(1 for decision in shared if decision.action == "trim")
        lines += [
            "",
            f"`ada share` would copy {len(shared)} of {len(self.decisions)} files "
            f"({_size(sum(decision.size for decision in shared))}) into share/"
            + (f", removing withheld columns from {trimmed}" if trimmed else "")
            + (", and add the original data file" if sharing.data == "raw" else "")
            + ".",
        ]
        return "\n".join(lines)


def review(study: Study | str | Path = "study.yml") -> Review:
    """Decide, for every file in the study's output folder, whether it may be shared and why."""
    if not isinstance(study, Study):
        study = load(study)
    folder = study.out_dir
    record = provenance.read_manifest(folder)
    if record is None:
        raise StudyError(
            [f"Nothing has been built in {study.output.dir}/ yet. Run `ada build` first."], study.path
        )
    notes = []
    original = None
    if study.sharing.withhold:
        try:
            original = read(study).original
        except StudyError as error:
            notes.append(
                "the data could not be read, so files were not searched for values of the withheld columns: "
                + "; ".join(error.problems)
            )
    rules = _Rules(study, record, original)
    if study.sharing.data in ("processed", "raw") and not study.sharing.row_level:
        notes.append(
            f"sharing.data is {study.sharing.data} but row_level is false, so the data files stay private; "
            "set row_level: true to share them."
        )
    decisions = [rules.decide(folder, rel) for rel in _files(folder)]
    return Review(study, folder, decisions, notes, rules)


def preview(study="study.yml") -> str:
    """What would be shared, file by file, and anything that should not be."""
    try:
        return review(study).text()
    except StudyError as error:
        if len(error.problems) == 1 and "Run `ada build` first" in error.problems[0]:
            return error.problems[0]
        raise


@dataclass
class Shared:
    """What ``ada share`` copied and left out, and whether the copy passed its final check."""

    ok: bool
    folder: Path
    copied: list[Decision] = field(default_factory=list)
    left_out: list[Decision] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    flagged: list[Decision] = field(default_factory=list)

    def text(self) -> str:
        if not self.ok:
            lines = [f"Refused: the copy in {self.folder} still held something withheld, so it was deleted."]
            lines += [f"  - {problem}" for problem in self.problems]
            return "\n".join(lines)
        size = sum(decision.size for decision in self.copied)
        lines = [f"Copied {len(self.copied)} files ({_size(size)}) into {self.folder}."]
        trimmed = [decision for decision in self.copied if decision.action == "trim"]
        if trimmed:
            lines += ["", "Withheld columns removed:"]
            lines += [f"  {decision.path}: {_names(decision.columns)}" for decision in trimmed]
        if self.left_out:
            lines += ["", "Left out:"]
            for decision in self.left_out:
                example = f" (such as “{decision.example}”)" if decision.example else ""
                lines.append(f"  {decision.path}: {decision.reason}{example}")
        checks = [
            (decision, flag) for decision in self.flagged for flag in decision.flags if "identifier" in flag
        ]
        if checks:
            lines += ["", "Check before publishing:"]
            lines += [f"  ! {decision.path}: {flag}" for decision, flag in checks]
        lines += [f"note: {note}" for note in self.notes]
        lines += ["", f"{SHARE_RECORD} in that folder lists what was left out, and why."]
        return "\n".join(lines)


def share(study: Study | str | Path = "study.yml", out: str | Path | None = None) -> Shared:
    """Copy the files that may be shared into ``out`` (``share/`` beside study.yml), then check the copy.

    Withheld columns are removed from every CSV; files left out are listed in
    SHARING.md. The copy is searched again with the same rules, and if anything
    withheld remains it is deleted and the result says what was found.
    """
    checked = review(study)
    study, source, rules = checked.study, checked.folder, checked.rules
    target = Path(out).resolve() if out else (study.root / "share").resolve()
    if target == source or source in target.parents or target in source.parents:
        raise StudyError(
            [f"The shared copy cannot go in {target}: it overlaps the output folder {source}."], study.path
        )
    _clear(target, study)
    _write(target / SHARE_RECORD, "# What was shared\n\nIncomplete: `ada share` did not finish.\n")
    copied, left_out = [], []
    for decision in checked.decisions:
        if decision.action == "leave":
            left_out.append(decision)
            continue
        destination = target / decision.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if decision.action == "trim":
            _write(destination, _without(source / decision.path, decision.columns))
        else:
            shutil.copyfile(source / decision.path, destination)
        copied.append(decision)
    notes = list(checked.notes)
    original = _share_original(study, target, notes)
    if original:
        (copied if original.action != "leave" else left_out).append(original)
    _write(target / SHARE_RECORD, share_record(study, copied, left_out))
    problems = []
    for rel in _files(target):
        decision = rules.decide(target, rel, rescan=True)
        if decision.action != "share":
            problems.append(
                f"{rel}: {decision.reason}" + (f"; {decision.flags[0]}" if decision.flags else "")
            )
    if problems:
        shutil.rmtree(target)
        return Shared(ok=False, folder=target, problems=problems)
    flagged = [decision for decision in checked.decisions if decision.flags and decision.action != "leave"]
    return Shared(ok=True, folder=target, copied=copied, left_out=left_out, notes=notes, flagged=flagged)


def _clear(target: Path, study: Study) -> None:
    if not target.exists():
        return
    if (target / SHARE_RECORD).is_file():
        shutil.rmtree(target)
    elif target.is_file() or any(target.iterdir()):
        raise StudyError(
            [
                f"{target} already exists and was not written by `ada share`. Remove it, or choose "
                "another folder with --out."
            ],
            study.path,
        )


def _share_original(study: Study, target: Path, notes: list[str]) -> Decision | None:
    """With ``sharing.data: raw``, the original data file, with any withheld columns removed."""
    if study.sharing.data != "raw":
        return None
    path = study.data_file
    rel = f"{DATA_FOLDER}/{path.name}"
    if not path.is_file():
        notes.append(f"the original data file {study.data_path} is not there, so it was not copied")
        return None
    if not study.sharing.row_level:
        return Decision(
            rel,
            path.stat().st_size,
            "leave",
            "original data file; it lists individual rows (sharing.row_level: false)",
        )
    destination = target / rel
    suffix = path.suffix.lower()
    header = _header(path) if suffix in (".csv", ".tsv") else None
    columns = tuple(name for name in study.sharing.withhold if name in (header or []))
    if (suffix in (".csv", ".tsv") and header is not None) or not study.sharing.withhold:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if columns:
            _write(destination, _without(path, columns))
            return Decision(rel, path.stat().st_size, "trim", "the original data file", columns=columns)
        shutil.copyfile(path, destination)
        return Decision(rel, path.stat().st_size, "share", "the original data file")
    return Decision(
        rel,
        path.stat().st_size,
        "leave",
        "the original data file; withheld columns can only be removed from a CSV or TSV file, so export "
        "one without them",
    )


def share_record(study: Study, copied: list[Decision], left_out: list[Decision]) -> str:
    """SHARING.md: the settings, and every file changed or left out. It holds no data values."""
    sharing = study.sharing
    lines = [
        "# What was shared",
        "",
        f"Copied by `ada share` from {study.output.dir}/ under the sharing settings in {study.path.name}: "
        f"`data: {sharing.data}`, `row_level: {str(sharing.row_level).lower()}`, "
        f"`withhold: [{', '.join(sharing.withhold)}]`. {len(copied)} files copied, {len(left_out)} left out.",
        "",
    ]
    trimmed = [decision for decision in copied if decision.action == "trim"]
    if trimmed:
        lines += ["## Withheld columns removed", ""]
        lines += [f"- {decision.path}: {_names(decision.columns)}" for decision in trimmed]
        lines.append("")
    if left_out:
        lines += ["## Left out", ""]
        lines += [f"- {decision.path}: {decision.reason}" for decision in left_out]
        lines.append("")
    return "\n".join(lines)
