"""``ada build``: make every asset of a study from ``study.yml`` and the data.

A build always starts from the original data file and rebuilds every asset, so
nothing can go stale between builds. It then compares itself with the previous
build's record and reports what changed in ``CHANGES.md``. It works in a scratch
folder beside the output folder and moves its files in only once every step has
succeeded, so a build that fails leaves the previous one as it was.

The output folder::

    paper-assets/
      README.md               what each file is, with captions
      CHANGES.md              what changed at each build, and why
      provenance.json         where every asset came from
      audit/                  data audit
      dictionary/             data dictionary
      processing/             processing record and the excluded rows
      tables/                 descriptive and results tables (CSV, Word, LaTeX)
      figures/                figures, their source data and captions
      methods/                methods draft built from what was run
      reproduce/              everything needed to rebuild the assets
      repository/             README, CITATION.cff and availability statements for a data or code repository
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import provenance
from .audit import audit, audit_markdown, findings_frame
from .descriptives import described_variables, descriptive_table, grouping, over_time_table
from .dictionary import dictionary_table
from .figures import write_captions
from .processing import SOURCE_ROW, Processed, read
from .provenance import Asset
from .study import Study, load
from .tables import Table, markdown, write

#: Folders the build owns and replaces at each build; anything else in the output folder is left alone.
OWNED = (
    "audit",
    "dictionary",
    "processing",
    "tables",
    "figures",
    "methods",
    "supplement",
    "reproduce",
    "repository",
)
#: Files the build owns at the top of the output folder.
TOP_FILES = ("README.md", provenance.CHANGES, provenance.MANIFEST)


@dataclass
class Build:
    """Everything one build made, handed from step to step."""

    study: Study
    processed: Processed
    out: Path
    assets: list[Asset] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    #: Analysis results by analysis id, for figures and the methods draft.
    results: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)
    #: sha256 of each custom script that was run.
    custom: dict[str, str] = field(default_factory=dict)
    changes: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: The resolved figure settings (``figures.Style``), set by the figures step.
    style: object = None

    @property
    def data(self) -> pd.DataFrame:
        return self.processed.data

    def folder(self, name: str) -> Path:
        path = self.out / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def rows(self, used: int | None = None) -> dict:
        excluded = sum(record.rows_removed for record in self.processed.records if record.kind == "exclude")
        steps = [record.number for record in self.processed.records if record.kind == "exclude"]
        return {
            "used": len(self.data) if used is None else used,
            "excluded": excluded,
            "exclusion_steps": steps,
        }

    def add_table(
        self,
        table: Table,
        *,
        section: str,
        depends: tuple[str, ...],
        calculation: str,
        code: str,
        rows: dict | None = None,
        numbers: dict | None = None,
        references: tuple[str, ...] = (),
        folder: str = "tables",
        row_level: bool = False,
    ) -> Asset:
        count = sum(1 for asset in self.assets if asset.kind == "table" and asset.section == section)
        if section == "main":
            table.number = f"Table {count + 1}"
        elif section == "supplement":
            table.number = f"Table S{count + 1}"
        files = write(table, self.folder(folder), self.study.output.tables)
        asset = Asset(
            id=table.id,
            kind="table",
            title=table.title,
            files=files,
            depends=(*depends, "output") if "output" not in depends else depends,
            calculation=calculation,
            code=code,
            rows=rows or self.rows(),
            numbers=numbers or {},
            references=references,
            number=table.number,
            caption=table.caption,
            section=section,
            row_level=row_level,
        )
        self.tables.append(table)
        self.assets.append(asset)
        return asset


def build(study: Study | str | Path = "study.yml", out: Path | None = None) -> Build:
    """Build every asset of ``study`` into its output folder (``output.dir``, or ``out``).

    The assets are made in a scratch folder beside the output folder and replace the previous
    build's only once every step has succeeded, so a build that stops (a changed data file, a
    custom function that raises) leaves the last good build as it was.
    """
    if not isinstance(study, Study):
        study = load(study)
    out = (out or study.out_dir).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    before = provenance.read_manifest(out)
    scratch = Path(tempfile.mkdtemp(prefix=f".{out.name}-building-", dir=out.parent))
    try:
        processed = read(study)
        result = Build(study=study, processed=processed, out=scratch)

        _audit(result)
        _dictionary(result)
        _processing(result)
        _descriptives(result)
        for step in _later_steps():
            step(result)
        write_captions(result)
        (scratch / "README.md").write_text(index_markdown(result), encoding="utf-8")

        record = provenance.manifest(study, scratch, result.assets, processed, result.custom)
        result.changes = provenance.compare(before, record)
        # This build's entry goes on top of the earlier ones.
        if (out / provenance.CHANGES).is_file():
            shutil.copyfile(out / provenance.CHANGES, scratch / provenance.CHANGES)
        provenance.write_changes(scratch, provenance.changes_markdown(result.changes, before, record))
        provenance.write_manifest(scratch, record)
        _move_in(scratch, out)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    for asset in result.assets:
        asset.files = [out / path.relative_to(scratch) for path in asset.files]
    result.out = out
    return result


def _move_in(scratch: Path, out: Path) -> None:
    """Replace the folders and files the build owns in ``out`` with the finished ones in ``scratch``.

    The previous ones are moved aside first, and back if a move fails, so ``out`` never holds
    part of each build. Anything else in ``out`` is left alone.
    """
    out.mkdir(exist_ok=True)
    previous = scratch / ".previous"
    previous.mkdir()
    names = [*OWNED, *TOP_FILES]
    aside, placed = [], []
    try:
        for name in names:
            if (out / name).exists():
                shutil.move(out / name, previous / name)
                aside.append(name)
        for name in names:
            if (scratch / name).exists():
                shutil.move(scratch / name, out / name)
                placed.append(name)
    except OSError:
        for name in placed:
            if (out / name).is_dir():
                shutil.rmtree(out / name)
            else:
                (out / name).unlink()
        for name in aside:
            shutil.move(previous / name, out / name)
        raise


def _later_steps():
    """The steps after the descriptive tables, each in its own module and run in this order."""
    from . import analyses, custom, figures, methods, package  # noqa: PLC0415

    return (analyses.run, figures.run, custom.run, methods.run, package.run)


DATA_SECTIONS = ("data", "variables", "processing", "design")


def _audit(result: Build) -> None:
    study = result.study
    result.findings = audit(study, result.processed)
    folder = result.folder("audit")
    report = folder / "data-audit.md"
    report.write_text(audit_markdown(study, result.findings), encoding="utf-8")
    table = folder / "data-audit.csv"
    findings_frame(result.findings).to_csv(table, index=False, lineterminator="\n")
    checks = sum(1 for finding in result.findings if finding.status == "check")
    result.assets.append(
        Asset(
            id="data-audit",
            kind="record",
            title="Data audit",
            files=[report, table],
            depends=(*DATA_SECTIONS, "decisions"),
            code="ada.research.audit.audit",
            rows=result.rows(),
            calculation="Counts of missing, unreadable, out-of-range and duplicated values; design checks; "
            "values more than 3.5 robust SDs (0.6745 × |x − median| / MAD) from the median.",
            references=(
                "Iglewicz, B. and Hoaglin, D. C. (1993). How to Detect and Handle Outliers. "
                "ASQC Quality Press.",
            ),
            numbers={"items_to_check": checks},
            section="record",
            row_level=True,
        )
    )


def _dictionary(result: Build) -> None:
    table = dictionary_table(result.study, result.processed)
    result.add_table(
        table,
        section="record",
        depends=("data", "variables", "processing", "sharing"),
        calculation="Observed ranges and level counts from the processed data.",
        code="ada.research.dictionary.dictionary_table",
        folder="dictionary",
    )


def _processing(result: Build) -> None:
    study, processed = result.study, result.processed
    folder = result.folder("processing")
    rows = [
        {
            "step": record.number,
            "kind": record.kind,
            "description": record.description,
            "reason": record.reason,
            "rows_before": record.rows_before,
            "rows_after": record.rows_after,
            "rows_removed": record.rows_removed,
            "values_changed": record.values_changed,
            "columns": "; ".join(record.columns),
            "removed_ids": "; ".join(record.removed_ids),
        }
        for record in processed.records
    ]
    log = folder / "processing-record.csv"
    pd.DataFrame(rows).to_csv(log, index=False, lineterminator="\n")
    lines = [
        f"# Processing record: {study.title}",
        "",
        f"The original file, {study.data_path} (sha256 {processed.sha256}), is never changed. These steps "
        f"turn its {len(processed.original):,} rows into the {len(processed.data):,} rows every asset uses.",
        "",
        "| Step | What | Reason | Rows before | Rows after | Values changed |",
        "|---|---|---|---|---|---|",
    ]
    for record in processed.records:
        lines.append(
            f"| {record.number} | {record.description} | {record.reason} | {record.rows_before:,} | "
            f"{record.rows_after:,} | {record.values_changed:,} |"
        )
    if processed.reading_notes:
        lines += ["", "What the reader did in step 1:", ""] + [
            f"- {note}" for note in processed.reading_notes
        ]
    files = [log]
    if len(processed.excluded):
        excluded = folder / "excluded-rows.csv"
        _ordered(processed.excluded).to_csv(excluded, index=False, lineterminator="\n")
        files.append(excluded)
        lines += [
            "",
            f"The {len(processed.excluded):,} excluded rows are in excluded-rows.csv, each with the step "
            f"that removed it and its line in the original file ({SOURCE_ROW}).",
        ]
    lines.append("")
    report = folder / "processing-record.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    files.insert(0, report)
    if study.sharing.data != "none":
        data_file = folder / "processed-data.csv"
        shared = processed.data.drop(columns=[c for c in study.sharing.withhold if c in processed.data])
        _ordered(shared).to_csv(data_file, index=False, lineterminator="\n")
        files.append(data_file)
    result.assets.append(
        Asset(
            id="processing-record",
            kind="record",
            title="Processing record",
            files=files,
            depends=("data", "processing", "variables", "sharing"),
            code="ada.research.processing.process",
            rows=result.rows(),
            numbers={"rows_read": len(processed.original), "rows_used": len(processed.data)},
            calculation="Each step applied in order to the rows left by the one before.",
            section="record",
            row_level=True,
        )
    )


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows in file order with categorical columns written as plain text."""
    frame = frame.sort_values(SOURCE_ROW, kind="stable") if SOURCE_ROW in frame else frame
    return frame.apply(
        lambda column: column.astype("string") if isinstance(column.dtype, pd.CategoricalDtype) else column
    )


def _descriptives(result: Build) -> None:
    study = result.study
    variables = described_variables(study)
    by = grouping(study)
    depends = (*DATA_SECTIONS, "descriptives")
    table = descriptive_table(study, result.processed)
    result.add_table(
        table,
        section="main",
        depends=depends,
        code="ada.research.descriptives.descriptive_table",
        calculation="Mean and SD (n − 1 denominator), median and quartiles (linear interpolation), "
        "counts and percentages of rows with a value" + (f", split by {by}" if by else "") + ".",
        numbers={"variables": len(variables)},
    )
    over_time = over_time_table(study, result.processed)
    if over_time is not None:
        result.add_table(
            over_time,
            section="main",
            depends=depends,
            code="ada.research.descriptives.over_time_table",
            calculation="Mean, SD and n of each outcome at each time point and group.",
        )


def index_markdown(result: Build) -> str:
    """The output folder's README: what each asset is, where its files are, and how to rebuild them."""
    study = result.study
    lines = [
        f"# {study.title}: paper assets",
        "",
        f"Built by ADA from {study.data_path} and study.yml. Every number traces back to the data through "
        "provenance.json; CHANGES.md says what changed at each build.",
        "",
        "Rebuild with `ada build`, and check a rebuild matches with `ada verify`.",
        "",
    ]
    for section, heading in (
        ("main", "For the paper"),
        ("supplement", "Supplementary material"),
        ("record", "Records"),
        ("package", "Sharing and reproduction"),
    ):
        chosen = [asset for asset in result.assets if asset.section == section]
        if not chosen:
            continue
        lines += [f"## {heading}", ""]
        for asset in chosen:
            files = ", ".join(
                f"[{path.name}]({path.relative_to(result.out).as_posix()})" for path in asset.files
            )
            heading = (
                f"{asset.number}. {asset.title}"
                if asset.number.startswith(("Table", "Figure"))
                else asset.title
            )
            lines.append(f"- **{heading}**: {files}")
        lines.append("")
    if result.tables:
        lines += ["## Tables", ""] + [
            markdown(table) for table in result.tables if table.number.startswith("Table")
        ]
    return "\n".join(lines)
