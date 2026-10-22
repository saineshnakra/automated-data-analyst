"""The data audit: what was read, what looks wrong, and whether the data fits the stated design.

The audit reports and never changes the data. Anything it finds that should
change a result has to become a step in ``processing`` with a reason, so the
change is visible in the processing record.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .processing import SOURCE_ROW, Processed, levels_of
from .study import Study

#: Robust z-score above which a value is listed as extreme (Iglewicz and Hoaglin, 1993).
EXTREME_Z = 3.5
#: Fewer clusters than this make a mixed model's variance estimates unreliable.
FEW_CLUSTERS = 5
#: Missing share above which a variable is marked for checking.
MISSING_SHARE = 0.2


@dataclass(frozen=True)
class Finding:
    area: str
    column: str
    #: "check" asks the researcher to look; "note" is for the record.
    status: str
    finding: str
    count: int = 0
    #: Lines in the original file (or subject ids) where it occurs, first few.
    where: str = ""


def _rows(frame: pd.DataFrame, limit: int = 10) -> str:
    lines = frame[SOURCE_ROW].tolist() if SOURCE_ROW in frame else []
    shown = ", ".join(map(str, lines[:limit]))
    return (
        f"lines {shown}" + (f" and {len(lines) - limit} more" if len(lines) > limit else "") if lines else ""
    )


def robust_z(values: pd.Series) -> pd.Series:
    """0.6745 × (x − median) / MAD, the modified z-score of Iglewicz and Hoaglin."""
    present = values.dropna()
    median = present.median()
    mad = (present - median).abs().median()
    if not mad or pd.isna(mad):
        return pd.Series(0.0, index=values.index)
    return 0.6745 * (values - median) / mad


def audit(study: Study, processed: Processed) -> list[Finding]:
    findings: list[Finding] = []
    add = findings.append
    original, data = processed.original, processed.data

    add(
        Finding(
            "file",
            "",
            "note",
            f"Read {study.data_path}: {len(original):,} rows and "
            f"{len(original.columns) - 1:,} columns. sha256 {processed.sha256}.",
            len(original),
        )
    )
    for note in processed.reading_notes:
        add(Finding("file", "", "note", note))
    if processed.undescribed:
        add(
            Finding(
                "columns",
                ", ".join(processed.undescribed),
                "note",
                f"{len(processed.undescribed)} column(s) are not described in study.yml and are not used.",
                len(processed.undescribed),
            )
        )
    for column, counts in processed.missing_codes.items():
        listed = ", ".join(f"{code} ({count:,})" for code, count in counts.items())
        add(
            Finding(
                "missing",
                column,
                "note",
                f"Missing-value codes read as missing: {listed}.",
                sum(counts.values()),
            )
        )
    for column, (count, examples) in processed.unreadable.items():
        add(
            Finding(
                "values",
                column,
                "check",
                f"{count:,} value(s) could not be read as {study.variable(column).type} and are treated as "
                f"missing, such as {', '.join(examples)}.",
                count,
            )
        )

    duplicated = original.drop(columns=[SOURCE_ROW]).duplicated(keep=False)
    if duplicated.any():
        add(
            Finding(
                "rows",
                "",
                "check",
                f"{int(duplicated.sum()):,} rows are exact copies of another row. They "
                "are kept; exclude them in `processing` if they are double entries.",
                int(duplicated.sum()),
                _rows(original[duplicated]),
            )
        )

    for name, variable in study.variables.items():
        if name not in original:
            continue
        series = data[name] if name in data else original[name]
        missing = int(series.isna().sum())
        if missing:
            share = missing / max(len(series), 1)
            add(
                Finding(
                    "missing",
                    name,
                    "check" if share > MISSING_SHARE else "note",
                    f"{missing:,} of {len(series):,} rows ({share:.0%}) have no value.",
                    missing,
                )
            )
        present = series.dropna()
        if len(present) and present.nunique() == 1:
            add(
                Finding(
                    "values",
                    name,
                    "check",
                    f"Every row has the same value ({present.iloc[0]}).",
                    len(present),
                )
            )
        if variable.type == "numeric" and variable.range:
            low, high = variable.range
            numbers = pd.to_numeric(series, errors="coerce")
            outside = ((numbers < low) if low is not None else False) | (
                (numbers > high) if high is not None else False
            )
            outside = pd.Series(outside, index=series.index).fillna(False)
            if outside.any():
                add(
                    Finding(
                        "values",
                        name,
                        "check",
                        f"{int(outside.sum()):,} value(s) fall outside the allowed range {low} to {high}.",
                        int(outside.sum()),
                        _rows(data.loc[outside[outside].index]),
                    )
                )
        if variable.type in ("categorical", "binary") and variable.levels:
            declared = {str(level) for level in variable.levels}
            unexpected = series.dropna().astype(str)
            unexpected = unexpected[~unexpected.isin(declared)]
            if len(unexpected):
                values = ", ".join(sorted(set(unexpected))[:8])
                add(
                    Finding(
                        "values",
                        name,
                        "check",
                        f"{len(unexpected):,} row(s) hold values not among the declared levels: {values}.",
                        len(unexpected),
                        _rows(data.loc[unexpected.index]),
                    )
                )
        if variable.type == "binary":
            levels = levels_of(data, name) if name in data else []
            if len(levels) > 2:
                add(
                    Finding(
                        "values",
                        name,
                        "check",
                        f"Declared binary but holds {len(levels)} values.",
                        len(levels),
                    )
                )
        if variable.role == "group" and name in data:
            counts = data[name].astype("string").value_counts()
            small = counts[counts < 3]
            if len(small):
                add(
                    Finding(
                        "design",
                        name,
                        "check",
                        "Groups with fewer than 3 rows: "
                        + ", ".join(f"{level} ({count})" for level, count in small.items())
                        + ".",
                        int(small.sum()),
                    )
                )
        if variable.type == "numeric" and variable.role in ("outcome", "covariate") and name in data:
            z = robust_z(pd.to_numeric(data[name], errors="coerce"))
            extreme = z.abs() > EXTREME_Z
            if extreme.any():
                if study.decisions.outliers == "flag":
                    add(
                        Finding(
                            "extremes",
                            name,
                            "check",
                            f"{int(extreme.sum()):,} value(s) lie more than {EXTREME_Z} robust SDs from the "
                            "median. They are kept (decisions.outliers: flag); exclude them in `processing` "
                            "only with a reason that does not depend on the result.",
                            int(extreme.sum()),
                            _rows(data.loc[extreme]),
                        )
                    )
                else:
                    add(
                        Finding(
                            "extremes",
                            name,
                            "note",
                            f"{int(extreme.sum()):,} value(s) lie more than {EXTREME_Z} robust SDs from the "
                            "median and are kept (decisions.outliers: keep).",
                            int(extreme.sum()),
                        )
                    )

    findings.extend(_design_findings(study, data))
    for record in processed.records:
        if record.kind == "exclude":
            add(
                Finding(
                    "processing",
                    ", ".join(record.columns),
                    "note",
                    f"Step {record.number} removed {record.rows_removed:,} row(s): {record.description}. "
                    f"Reason: {record.reason}",
                    record.rows_removed,
                )
            )
    add(Finding("processing", "", "note", f"{len(data):,} rows remain for the analyses.", len(data)))
    return findings


def _design_findings(study: Study, data: pd.DataFrame) -> list[Finding]:
    findings = []
    add = findings.append
    subject, time, cluster = study.subject, study.time, study.cluster
    groups = study.with_role("group")
    if subject and subject in data:
        per_subject = data[subject].value_counts()
        if study.design == "independent" and (per_subject > 1).any():
            repeated = per_subject[per_subject > 1]
            add(
                Finding(
                    "design",
                    subject,
                    "check",
                    f"The design is independent, but {len(repeated):,} subject(s) appear in more than one "
                    f"row (up to {int(repeated.max())}). If they were measured more than once, set "
                    "`design.type: repeated`.",
                    len(repeated),
                )
            )
        if study.design == "paired":
            condition = next((name for name in groups if name in data), None)
            if condition:
                levels = levels_of(data, condition)
                if len(levels) != 2:
                    add(
                        Finding(
                            "design",
                            condition,
                            "check",
                            f"A paired design needs exactly two conditions; {condition} has {len(levels)}. "
                            "Use `design.type: repeated` for three or more.",
                            len(levels),
                        )
                    )
                else:
                    seen = data.groupby(subject, observed=True)[condition].nunique()
                    single = seen[seen < 2]
                    add(
                        Finding(
                            "design",
                            subject,
                            "check" if len(single) else "note",
                            f"{int((seen == 2).sum()):,} subject(s) have both conditions"
                            + (
                                f"; {len(single):,} have only one and cannot be paired: "
                                + ", ".join(map(str, single.index[:10]))
                                if len(single)
                                else ""
                            )
                            + ".",
                            len(single),
                        )
                    )
                keys = data[[subject, condition]].dropna()
                doubled = keys.duplicated(keep=False)
                if doubled.any():
                    add(
                        Finding(
                            "design",
                            subject,
                            "check",
                            f"{int(doubled.sum()):,} rows repeat a subject and condition. Average them or "
                            "exclude extras before a paired analysis.",
                            int(doubled.sum()),
                        )
                    )
        if study.design == "repeated":
            add(
                Finding(
                    "design",
                    subject,
                    "note",
                    f"{per_subject.size:,} subjects with {int(per_subject.min())} to "
                    f"{int(per_subject.max())} rows each.",
                    per_subject.size,
                )
            )
            single = per_subject[per_subject == 1]
            if len(single):
                add(
                    Finding(
                        "design",
                        subject,
                        "note",
                        f"{len(single):,} subject(s) have one row only; a mixed model still uses them.",
                        len(single),
                    )
                )
            if time and time in data:
                doubled = data[[subject, time]].dropna().duplicated(keep=False)
                if doubled.any():
                    add(
                        Finding(
                            "design",
                            time,
                            "check",
                            f"{int(doubled.sum()):,} rows repeat a subject at the same {time}.",
                            int(doubled.sum()),
                        )
                    )
                per_time = data.groupby(time, observed=True)[subject].nunique()
                add(
                    Finding(
                        "design",
                        time,
                        "note",
                        "Subjects measured at each time point: "
                        + ", ".join(f"{moment} ({count})" for moment, count in per_time.items())
                        + ".",
                        int(per_time.sum()),
                    )
                )
    if study.design == "clustered" and cluster and cluster in data:
        per_cluster = data[cluster].value_counts()
        add(
            Finding(
                "design",
                cluster,
                "check" if per_cluster.size < FEW_CLUSTERS else "note",
                f"{per_cluster.size:,} clusters with {int(per_cluster.min())} to "
                f"{int(per_cluster.max())} rows each."
                + (
                    f" With fewer than {FEW_CLUSTERS} clusters the between-cluster variance is poorly "
                    "estimated; report the analysis with that caveat."
                    if per_cluster.size < FEW_CLUSTERS
                    else ""
                ),
                per_cluster.size,
            )
        )
        for name in groups:
            if name in data:
                mixed = data.groupby(cluster, observed=True)[name].nunique()
                if (mixed > 1).any() and (mixed == 1).any():
                    add(
                        Finding(
                            "design",
                            name,
                            "note",
                            f"{name} varies within some clusters and not others.",
                            int((mixed > 1).sum()),
                        )
                    )
    return findings


def findings_frame(findings: list[Finding]) -> pd.DataFrame:
    return pd.DataFrame(
        [finding.__dict__ for finding in findings],
        columns=["area", "column", "status", "finding", "count", "where"],
    )


def audit_markdown(study: Study, findings: list[Finding]) -> str:
    checks = [finding for finding in findings if finding.status == "check"]
    lines = [f"# Data audit: {study.title}", ""]
    lines.append(
        f"{len(checks)} item(s) to check."
        if checks
        else "Nothing to check; the notes below are for the record."
    )
    lines.append("")
    for heading, status in (("To check", "check"), ("Notes", "note")):
        chosen = [finding for finding in findings if finding.status == status]
        if not chosen:
            continue
        lines += [f"## {heading}", ""]
        for finding in chosen:
            column = f"**{finding.column}**: " if finding.column else ""
            where = f" ({finding.where})" if finding.where else ""
            lines.append(f"- {column}{finding.finding}{where}")
        lines.append("")
    lines += [
        "The audit reports and changes nothing. To act on an item, add a step to `processing` in study.yml "
        "with its reason; it then appears in the processing record.",
        "",
    ]
    return "\n".join(lines)


def has_problems(findings: list[Finding]) -> bool:
    return any(finding.status == "check" for finding in findings)
