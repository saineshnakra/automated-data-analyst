"""Descriptive tables: each variable summarised overall and by group, and outcomes over time."""

from __future__ import annotations

import pandas as pd

from .processing import Processed, levels_of
from .study import Study
from .tables import Table, decimals_for, number

STATISTICS = ["n", "missing", "mean", "sd", "median", "q1", "q3", "min", "max", "count", "percent"]


def described_variables(study: Study) -> list[str]:
    """The variables in the descriptive table: those listed, else outcomes, then covariates and groups."""
    if study.descriptives:
        return list(study.descriptives)
    by = grouping(study)
    ordered = study.with_role("outcome") + study.with_role("covariate") + study.with_role("group")
    return [name for name in dict.fromkeys(ordered) if name != by and study.variable(name).type != "text"]


def grouping(study: Study) -> str | None:
    if study.descriptives_by:
        return study.descriptives_by
    groups = study.with_role("group")
    return groups[0] if groups else None


def _numeric_rows(values: pd.Series) -> dict:
    present = values.dropna()
    return {
        "n": int(present.size),
        "missing": int(values.isna().sum()),
        "mean": present.mean() if present.size else None,
        "sd": present.std(ddof=1) if present.size > 1 else None,
        "median": present.median() if present.size else None,
        "q1": present.quantile(0.25) if present.size else None,
        "q3": present.quantile(0.75) if present.size else None,
        "min": present.min() if present.size else None,
        "max": present.max() if present.size else None,
    }


def descriptive_table(study: Study, processed: Processed) -> Table:
    data = processed.data
    by = grouping(study)
    columns: list[tuple[str, pd.Series]] = [("Overall", pd.Series(True, index=data.index))]
    if by:
        for level in levels_of(data, by):
            columns.append((level, (data[by].astype("string") == level).fillna(False)))
    long_rows = []
    display_rows = []

    def header(label: str, mask: pd.Series) -> str:
        return f"{label} (n = {int(mask.sum()):,})"

    if study.subject and study.subject in data:
        cells = [data.loc[mask, study.subject].nunique() for _, mask in columns]
        display_rows.append(["Subjects", "n", *[f"{count:,}" for count in cells]])
        for (label, _), count in zip(columns, cells, strict=True):
            long_rows.append(
                {
                    "variable": study.subject,
                    "level": "",
                    "group": label,
                    "count": count,
                    "statistic": "subjects",
                }
            )

    for name in described_variables(study):
        if name not in data:
            continue
        variable = study.variable(name)
        series = data[name]
        if variable.type in ("categorical", "binary"):
            levels = levels_of(data, name)
            display_rows.append([variable.display, "n (%)", *["" for _ in columns]])
            for level in levels:
                cells = []
                for label, mask in columns:
                    subset = series[mask]
                    count = int((subset.astype("string") == level).sum())
                    present = int(subset.notna().sum())
                    percent = 100 * count / present if present else None
                    cells.append(f"{count:,} ({number(percent, 1)})")
                    long_rows.append(
                        {
                            "variable": name,
                            "level": level,
                            "group": label,
                            "count": count,
                            "percent": percent,
                            "n": present,
                            "statistic": "count",
                        }
                    )
                display_rows.append([f"  {level}", "", *cells])
            missing = [int(series[mask].isna().sum()) for _, mask in columns]
            if any(missing):
                display_rows.append(["  Missing", "n", *[f"{count:,}" for count in missing]])
            continue
        if variable.type != "numeric":
            continue
        decimals = decimals_for(series, variable.decimals)
        summaries = []
        for label, mask in columns:
            summary = _numeric_rows(series[mask])
            summaries.append(summary)
            long_rows.append(
                {"variable": name, "level": "", "group": label, "statistic": "numeric", **summary}
            )
        display_rows.append(
            [
                variable.display,
                "Mean (SD)",
                *[f"{number(s['mean'], decimals)} ({number(s['sd'], decimals)})" for s in summaries],
            ]
        )
        display_rows.append(
            [
                "",
                "Median [IQR]",
                *[
                    f"{number(s['median'], decimals)} "
                    f"[{number(s['q1'], decimals)}, {number(s['q3'], decimals)}]"
                    for s in summaries
                ],
            ]
        )
        display_rows.append(
            [
                "",
                "Range",
                *[f"{number(s['min'], decimals)} to {number(s['max'], decimals)}" for s in summaries],
            ]
        )
        if any(s["missing"] for s in summaries):
            display_rows.append(["", "Missing, n", *[f"{s['missing']:,}" for s in summaries]])

    display = pd.DataFrame(
        display_rows, columns=["Variable", "", *[header(label, mask) for label, mask in columns]]
    )
    data_frame = pd.DataFrame(long_rows)
    for column in ["variable", "level", "group", "statistic", *STATISTICS]:
        if column not in data_frame:
            data_frame[column] = None
    data_frame = data_frame[["variable", "level", "group", "statistic", *STATISTICS]]
    notes = [f"n counts rows; each row is {study.observation}."]
    if by:
        notes.append(f"Columns split the rows by {study.label(by)}.")
    notes.append(
        "SD, standard deviation; IQR, interquartile range (25th to 75th percentile). "
        "Percentages are of rows with a value."
    )
    if processed.excluded is not None and len(processed.excluded):
        notes.append(
            f"After the exclusions in the processing record ({len(processed.excluded):,} rows removed)."
        )
    title = "Characteristics of the data" + (f" by {study.label(by)}" if by else "")
    return Table(id="descriptives", title=title, display=display, data=data_frame, notes=notes)


def over_time_table(study: Study, processed: Processed) -> Table | None:
    """Each outcome's mean (SD) and n at each time point, by group: the shape of a repeated-measures study."""
    time = study.time
    data = processed.data
    if not time or time not in data:
        return None
    by = grouping(study)
    outcomes = [
        name for name in study.with_role("outcome") if study.variable(name).type == "numeric" and name in data
    ]
    if not outcomes:
        return None
    groups = levels_of(data, by) if by else ["All"]
    times = (
        levels_of(data, time)
        if isinstance(data[time].dtype, pd.CategoricalDtype)
        else sorted(data[time].dropna().unique())
    )
    long_rows, display_rows = [], []
    for name in outcomes:
        decimals = decimals_for(data[name], study.variable(name).decimals)
        for moment in times:
            at_time = data[time] == moment
            cells = []
            for group in groups:
                mask = at_time & ((data[by].astype("string") == group).fillna(False) if by else True)
                values = data.loc[mask, name].dropna()
                mean = values.mean() if len(values) else None
                sd = values.std(ddof=1) if len(values) > 1 else None
                long_rows.append(
                    {
                        "variable": name,
                        "time": moment,
                        "group": group,
                        "n": int(len(values)),
                        "mean": mean,
                        "sd": sd,
                    }
                )
                cells.append(f"{number(mean, decimals)} ({number(sd, decimals)}), {len(values)}")
            display_rows.append([study.variable(name).display, str(moment), *cells])
    display = pd.DataFrame(display_rows, columns=["Outcome", study.label(time), *groups])
    display.loc[display["Outcome"].duplicated(), "Outcome"] = ""
    notes = ["Cells show mean (SD), n."]
    return Table(
        id="outcomes-over-time",
        title=f"Outcomes at each {study.label(time).lower()}" + (f" by {study.label(by)}" if by else ""),
        display=display,
        data=pd.DataFrame(long_rows),
        notes=notes,
    )
