"""Synthetic demo data: a table shaped like the study's processed data, every value drawn at random.

When ``sharing.data`` is ``demo`` the real data stay private, and the
reproduction package carries this table instead so that anyone can run the
whole build. Each column is drawn on its own from its observed distribution, so
no relationship between columns and no real row survives. The design's
skeleton is kept: each subject or cluster keeps its rows and time points, under
a new code.
"""

from __future__ import annotations

import copy
import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..numeric import to_numbers
from .processing import SOURCE_ROW
from .study import Study

#: Seed for the draws, so every build writes the same file.
SEED = 2718
#: Decimal places kept at most when a drawn number is rounded like the observed ones.
MAX_DECIMALS = 6

PERMUTED = (
    "the observed values in random order (numpy.random.Generator.permutation), so each level keeps its count"
)
QUANTILES = (
    "drawn from the observed distribution by inverse-transform sampling (numpy.quantile with linear "
    "interpolation of uniform draws)"
)


@dataclass
class Synthetic:
    data: pd.DataFrame
    #: How each column was made, in words, for demo-data.README.md.
    methods: dict[str, str] = field(default_factory=dict)
    #: The column whose rows were kept together (the subject or cluster), and how many there are.
    unit: str | None = None
    units: int = 0
    #: Withheld columns, which are not in the table.
    left_out: tuple[str, ...] = ()
    seed: int = SEED


def synthesize(study: Study, data: pd.DataFrame, seed: int = SEED) -> Synthetic:
    """Draw a table with the columns, types and levels of ``data`` and none of its values."""
    unit = _unit(study, data)
    withheld = [name for name in study.sharing.withhold if name in data and name != unit]
    frame = data.drop(columns=[SOURCE_ROW, *withheld], errors="ignore").reset_index(drop=True)
    codes = pd.factorize(frame[unit], use_na_sentinel=False)[0] if unit else None
    kept = _structure(study, frame, unit, codes)
    columns, methods = {}, {}
    for name in frame.columns:
        rng = np.random.default_rng([seed, zlib.crc32(str(name).encode("utf-8"))])
        series = frame[name]
        variable = study.variables.get(name)
        kind = variable.type if variable else _kind_of(series)
        if name == unit or kind == "text" or (variable is not None and variable.role == "identifier"):
            columns[name], methods[name] = _codes(series, name)
            if name == unit and name in study.sharing.withhold:
                methods[name] += "; withheld, but kept as new codes because the design needs it"
        elif name in kept:
            columns[name], methods[name] = series, "kept as it is: part of the design"
        elif codes is not None and _constant_within(series, codes):
            per_unit = series.groupby(codes, sort=True).first().reset_index(drop=True)
            drawn, how = _draw(per_unit, kind, rng)
            columns[name] = drawn.iloc[codes].reset_index(drop=True)
            methods[name] = f"{how}; one value per {unit}"
        else:
            columns[name], methods[name] = _draw(series, kind, rng)
    return Synthetic(
        data=pd.DataFrame(columns),
        methods=methods,
        unit=unit,
        units=int(codes.max()) + 1 if codes is not None and len(codes) else 0,
        left_out=tuple(withheld),
        seed=seed,
    )


def _unit(study: Study, data: pd.DataFrame) -> str | None:
    """The column that groups rows in the design: the subject, or the cluster."""
    name = {"paired": study.subject, "repeated": study.subject, "clustered": study.cluster}.get(study.design)
    return name if name and name in data else None


def _structure(study: Study, frame: pd.DataFrame, unit: str | None, codes) -> set[str]:
    """Columns kept as they are: the time point, and conditions that change within a subject or cluster."""
    if unit is None:
        return set()
    kept = {study.time} if study.time in frame else set()
    for name in study.with_role("group"):
        if name in frame and not _constant_within(frame[name], codes):
            kept.add(name)
    return kept


def _constant_within(series: pd.Series, codes) -> bool:
    return bool((series.groupby(codes).nunique(dropna=False) <= 1).all())


def _kind_of(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    return "text"


def _codes(series: pd.Series, name: str) -> tuple[pd.Series, str]:
    """New codes in place of each distinct value, so rows that shared a value still share one."""
    codes, uniques = pd.factorize(series)
    width = max(2, len(str(len(uniques))))
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        labels = pd.array(np.where(codes >= 0, codes + 1, 0), dtype="Int64")
        labels[codes < 0] = pd.NA
        return pd.Series(labels), f"replaced by new numbers 1 to {len(uniques)}"
    initial = next((character.upper() for character in str(name) if character.isalnum()), "X")
    names = np.array([f"demo-{initial}{number:0{width}d}" for number in range(1, len(uniques) + 1)] + [None])
    values = pd.Series(names[codes], dtype="string")
    return (
        values,
        f"replaced by new codes demo-{initial}{1:0{width}d} to demo-{initial}{len(uniques):0{width}d}",
    )


def _draw(series: pd.Series, kind: str, rng: np.random.Generator) -> tuple[pd.Series, str]:
    if kind == "date" or pd.api.types.is_datetime64_any_dtype(series):
        return _draw_dates(series, rng)
    if kind == "numeric" and pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        values, decimals = _quantile_draws(to_numbers(series).astype(float), rng)
        drawn = pd.Series(values)
        if decimals == 0:
            drawn = drawn.round().astype("Int64")
        return drawn, f"{QUANTILES}, rounded to {decimals} decimal place{'s' if decimals != 1 else ''}"
    order = rng.permutation(len(series))
    return series.iloc[order].reset_index(drop=True), PERMUTED


def _quantile_draws(numbers: pd.Series, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Uniform draws mapped through the observed quantiles, with as many missing values as observed."""
    size = len(numbers)
    present = numbers.dropna().to_numpy()
    if present.size == 0:
        return np.full(size, np.nan), 0
    decimals = _decimals(present)
    values = np.round(np.quantile(present, rng.uniform(size=size), method="linear"), decimals)
    missing = size - present.size
    if missing:
        values[rng.choice(size, missing, replace=False)] = np.nan
    return values, decimals


def _decimals(values: np.ndarray) -> int:
    for places in range(MAX_DECIMALS + 1):
        if np.allclose(values, np.round(values, places), rtol=0, atol=1e-9):
            return places
    return MAX_DECIMALS


def _draw_dates(series: pd.Series, rng: np.random.Generator) -> tuple[pd.Series, str]:
    dates = pd.to_datetime(series, errors="coerce")
    nanoseconds = dates.astype("int64").where(dates.notna()).astype(float)
    values, _ = _quantile_draws(nanoseconds, rng)
    drawn = pd.to_datetime(pd.Series(values), unit="ns")
    present = dates.dropna()
    if present.empty or (present == present.dt.normalize()).all():
        return drawn.dt.strftime("%Y-%m-%d").astype("string"), f"{QUANTILES}, to the day"
    return drawn.dt.strftime("%Y-%m-%dT%H:%M:%S").astype("string"), f"{QUANTILES}, to the second"


def demo_study(
    study: Study, synthetic: Synthetic, data_name: str, sha256: str, output: str
) -> tuple[str, list]:
    """``study.yml`` pointed at the demo data, with what the demo data can't support taken out.

    The processing steps go, because the draws already follow the processed
    data. Analyses, figures and descriptive settings that name a withheld column
    go too, and each change is returned in words for the README.
    """
    import yaml  # noqa: PLC0415

    raw = copy.deepcopy(study.raw)
    gone = set(synthetic.left_out)
    changes = [f"`data.path` is {data_name}, with its sha256"]
    section = dict(raw.get("study") or {})
    section["title"] = f"{study.title} (synthetic demo data)"
    raw["study"] = section
    data = {"path": data_name, "sha256": sha256, "observation": study.observation}
    if study.missing_codes:
        data["missing_codes"] = list(study.missing_codes)
    raw["data"] = data
    if study.steps:
        changes.append(
            f"the {len(study.steps)} processing step{'s were' if len(study.steps) != 1 else ' was'} removed, "
            "because the synthetic values were drawn from the processed data"
        )
    raw["processing"] = []
    if gone:
        raw["variables"] = {
            name: entry for name, entry in (raw.get("variables") or {}).items() if str(name) not in gone
        }
        names = ", ".join(sorted(gone))
        changes.append(
            f"the withheld columns {names} were left out"
            if len(gone) > 1
            else f"the withheld column {names} was left out"
        )
        design = dict(raw.get("design") or {})
        for key in ("subject", "cluster", "time"):
            if design.get(key) in gone:
                design.pop(key)
        raw["design"] = design
    kept_analyses = []
    for analysis in study.analyses:
        named = {analysis.outcome, analysis.predictor, analysis.by, *analysis.covariates} & gone
        if named:
            changes.append(f"analysis {analysis.id} was removed: it uses {', '.join(sorted(named))}")
        else:
            kept_analyses.append(analysis.id)
    if raw.get("analyses"):
        raw["analyses"] = [item for item in raw["analyses"] if str(item.get("id")) in kept_analyses]
    if raw.get("figures"):
        kept_figures = []
        for item in raw["figures"]:
            named = {item.get(key) for key in ("x", "y", "by")} & gone
            lost = item.get("analysis") and str(item["analysis"]) not in kept_analyses
            if named or lost:
                changes.append(
                    f"figure {item.get('id')} was removed: it "
                    + ("uses a withheld column" if named else "draws a removed analysis")
                )
            else:
                kept_figures.append(item)
        ids = {str(item.get("id")) for item in kept_figures}
        for item in kept_figures:
            if item.get("panels"):
                item["panels"] = [panel for panel in item["panels"] if str(panel) in ids]
        raw["figures"] = kept_figures
    descriptives = raw.get("descriptives")
    if isinstance(descriptives, dict) and gone:
        if descriptives.get("by") in gone:
            descriptives.pop("by")
        if isinstance(descriptives.get("variables"), list):
            descriptives["variables"] = [name for name in descriptives["variables"] if name not in gone]
    output_section = dict(raw.get("output") or {})
    output_section["dir"] = output
    raw["output"] = output_section
    changes.append(f"`output.dir` is {output}")
    raw["sharing"] = {"data": "processed", "withhold": [], "row_level": True}
    header = [
        "# SYNTHETIC DEMO STUDY, written by ADA from study.yml. It runs the same build on",
        f"# {data_name}, a synthetic table, so none of its results are the study's.",
        f"# {data_name.removesuffix('.csv')}.README.md says how the table was made. Changed from study.yml:",
        *[f"#   - {change}" for change in changes],
        "",
    ]
    body = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, width=100)
    return "\n".join(header) + body, changes
