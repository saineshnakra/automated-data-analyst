"""The data dictionary: what each column means, its unit and type, and what it holds."""

from __future__ import annotations

import pandas as pd

from .processing import Processed, levels_of
from .study import Study
from .tables import Table, decimals_for, number


def _allowed(variable) -> str:
    if variable.levels:
        return ", ".join(map(str, variable.levels))
    if variable.range:
        low, high = variable.range
        return f"{'' if low is None else low} to {'' if high is None else high}".strip()
    return ""


def _observed(series: pd.Series, variable) -> str:
    present = series.dropna()
    if present.empty:
        return "no values"
    if variable.type == "numeric":
        decimals = decimals_for(present, variable.decimals)
        return f"{number(present.min(), decimals)} to {number(present.max(), decimals)}"
    if variable.type == "date":
        return f"{present.min():%Y-%m-%d} to {present.max():%Y-%m-%d}"
    if variable.type in ("categorical", "binary"):
        frame = series.to_frame()
        counts = series.astype("string").value_counts()
        levels = levels_of(frame, series.name)
        shown = [f"{level} ({int(counts.get(level, 0)):,})" for level in levels[:12]]
        return ", ".join(shown) + (f", and {len(levels) - 12} more" if len(levels) > 12 else "")
    return f"{present.nunique():,} distinct values"


def dictionary_table(study: Study, processed: Processed) -> Table:
    data = processed.data
    rows, display = [], []
    withheld = set(study.sharing.withhold)
    for name, variable in study.variables.items():
        series = data[name] if name in data else pd.Series(dtype=float, name=name)
        codes = [*map(str, study.missing_codes), *map(str, variable.missing_codes)]
        role = variable.role
        if name == study.subject:
            role = "identifier (subject)"
        elif name == study.cluster:
            role = "identifier (cluster)"
        elif name == study.time:
            role = "time"
        shared = "withheld" if name in withheld else "yes"
        row = {
            "variable": name,
            "label": variable.label or name,
            "role": role,
            "type": variable.type,
            "unit": variable.unit,
            "description": variable.description,
            "allowed": _allowed(variable),
            "observed": _observed(series, variable),
            "present": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "missing_codes": "; ".join(codes),
            "shared": shared,
        }
        rows.append(row)
        display.append(
            [
                name,
                row["label"],
                role,
                variable.type,
                variable.unit or "–",
                variable.description or "–",
                row["allowed"] or "–",
                row["observed"],
                f"{row['present']:,}",
                f"{row['missing']:,}",
                shared,
            ]
        )
    frame = pd.DataFrame(
        display,
        columns=[
            "Column",
            "Label",
            "Role",
            "Type",
            "Unit",
            "Description",
            "Allowed",
            "Observed",
            "Present",
            "Missing",
            "Shared",
        ],
    )
    notes = [
        f"One row is {study.observation}. Observed values, present and missing counts are for the "
        f"{len(data):,} rows left after processing.",
        "Allowed values come from study.yml; the data audit lists any value outside them.",
    ]
    if processed.undescribed:
        notes.append("Not described in study.yml and not used: " + ", ".join(processed.undescribed) + ".")
    return Table(
        id="data-dictionary", title="Data dictionary", display=frame, data=pd.DataFrame(rows), notes=notes
    )
