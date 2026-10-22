"""Read the study's data and apply its processing steps, recording each one.

The original file is never changed. Every step from reading onwards is written
to the processing record with the rows before and after, what it changed, and
the reason given in ``study.yml``; removed rows are kept, with the step that
removed them and their line in the original file, so any exclusion can be
checked and undone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..analysis import clean_dataframe
from ..file_io import read_tabular_file
from .study import Condition, Step, Study, StudyError, sha256_of

#: Column added to every row: its line in the original file (the header is line 1).
SOURCE_ROW = "source_row"


@dataclass(frozen=True)
class Record:
    """One line of the processing record."""

    number: int
    kind: str
    description: str
    reason: str
    rows_before: int
    rows_after: int
    #: Values set or changed by the step, for steps that change values rather than rows.
    values_changed: int = 0
    columns: tuple[str, ...] = ()
    #: Identifier values of the removed rows, when the design names a subject.
    removed_ids: tuple[str, ...] = ()

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after


@dataclass
class Processed:
    """The data at each stage, with the record of how one became the next."""

    #: The table as read, before any study rule. Kept to show what each step did.
    original: pd.DataFrame
    #: The table every asset is built from.
    data: pd.DataFrame
    records: list[Record] = field(default_factory=list)
    #: Every removed row as it was when removed, with the step number and reason.
    excluded: pd.DataFrame = field(default_factory=pd.DataFrame)
    sha256: str = ""
    #: Columns in the file that ``study.yml`` doesn't describe. They are kept but unused.
    undescribed: tuple[str, ...] = ()
    #: Per column, how many values each missing-value code turned into missing.
    missing_codes: dict[str, dict[str, int]] = field(default_factory=dict)
    #: Per column, values that could not be read as the declared type, with examples.
    unreadable: dict[str, tuple[int, tuple[str, ...]]] = field(default_factory=dict)
    reading_notes: tuple[str, ...] = ()

    def rows_used_by(self, columns: list[str]) -> pd.Series:
        """The rows where every one of ``columns`` has a value: the complete cases for an analysis."""
        present = [column for column in columns if column in self.data]
        return self.data[present].notna().all(axis=1) if present else pd.Series(True, index=self.data.index)


def read(study: Study) -> Processed:
    """Read the data file, check it against ``data.sha256`` and apply the study's steps."""
    path = study.data_file
    if not path.is_file():
        raise StudyError([f"The data file {study.data_path} is not there (looked in {path})."], study.path)
    digest = sha256_of(path)
    if study.data_sha256 and study.data_sha256 != digest:
        raise StudyError(
            [
                f"{study.data_path} has changed since study.yml recorded it: its sha256 is now {digest}, "
                f"not {study.data_sha256}. If the change is intended, put the new value in `data.sha256`; "
                "otherwise restore the original file."
            ],
            study.path,
        )
    try:
        raw = read_tabular_file(path.read_bytes(), path.name, study.data_sheet)
    except ValueError as error:
        raise StudyError([f"{study.data_path} could not be read: {error}"], study.path) from error
    raw.insert(0, SOURCE_ROW, np.arange(2, len(raw) + 2))
    return process(study, raw, digest)


def process(study: Study, raw: pd.DataFrame, digest: str = "") -> Processed:
    """Apply reading, missing-value codes, types and the processing steps to ``raw``."""
    records: list[Record] = []
    source_rows = raw[SOURCE_ROW]
    cleaned, report = clean_dataframe(raw.drop(columns=[SOURCE_ROW]))
    cleaned.insert(0, SOURCE_ROW, source_rows.loc[cleaned.index].to_numpy())
    records.append(
        Record(
            number=1,
            kind="read",
            description=f"Read {study.data_path}"
            + (f", sheet {study.data_sheet}" if study.data_sheet else ""),
            reason="ADA's reader parses numbers, dates and missing-value markers and drops empty rows "
            "and columns.",
            rows_before=len(raw),
            rows_after=len(cleaned),
            values_changed=_set_missing(raw, cleaned),
            columns=(),
        )
    )
    missing = [
        name
        for name in study.variables
        if name not in cleaned.columns and not any(step.to == name for step in study.steps)
    ]
    if missing:
        raise StudyError(
            [
                f"study.yml describes {name!r}, but {study.data_path} has no such column. Columns read: "
                f"{', '.join(map(str, cleaned.columns[1:]))}."
                for name in missing
            ],
            study.path,
        )
    processed = Processed(
        original=cleaned.copy(),
        data=cleaned,
        records=records,
        sha256=digest,
        reading_notes=tuple(report.notes),
    )
    processed.undescribed = tuple(
        column for column in cleaned.columns if column != SOURCE_ROW and column not in study.variables
    )
    _apply_missing_codes(study, processed)
    _apply_types(study, processed)
    excluded = []
    for step in study.steps:
        before = processed.data
        if step.kind == "exclude":
            mask = _matches(study, step, before)
            removed = before.loc[mask].copy()
            removed.insert(1, "removed_by_step", len(processed.records) + 1)
            removed.insert(2, "reason", step.reason)
            excluded.append(removed)
            processed.data = before.loc[~mask].copy()
            ids = ()
            id_column = step.id_column or study.subject
            if id_column and id_column in removed:
                ids = tuple(dict.fromkeys(map(str, removed[id_column].dropna())))
            _record(processed, step, before, values=0, ids=ids)
        elif step.kind == "transform":
            processed.data, changed = _transform(step, before)
            _record(processed, step, before, values=changed)
        else:
            processed.data, changed = _recode(step, before)
            _record(processed, step, before, values=changed)
    if excluded:
        processed.excluded = pd.concat(excluded)
    _order_levels(study, processed)
    return processed


def _set_missing(raw: pd.DataFrame, cleaned: pd.DataFrame) -> int:
    """Values the reader set to missing, such as "n/a" or a stray -99, in the rows it kept."""
    shared = [column for column in cleaned.columns if column in raw.columns and column != SOURCE_ROW]
    kept = raw.loc[cleaned.index, shared]
    return int((kept.notna() & cleaned[shared].isna()).to_numpy().sum())


def _record(processed: Processed, step: Step, before: pd.DataFrame, *, values: int, ids=()) -> None:
    columns = tuple(
        dict.fromkeys(
            [condition.column for condition in step.where] + [name for name in (step.column, step.to) if name]
        )
    )
    processed.records.append(
        Record(
            number=len(processed.records) + 1,
            kind=step.kind,
            description=step.describe(),
            reason=step.reason,
            rows_before=len(before),
            rows_after=len(processed.data),
            values_changed=values,
            columns=columns,
            removed_ids=ids,
        )
    )


def _code_matches(series: pd.Series, code) -> pd.Series:
    """Rows of ``series`` holding ``code``, comparing numbers as numbers and text as trimmed text."""
    if isinstance(code, bool):
        return pd.Series(False, index=series.index)
    if pd.api.types.is_numeric_dtype(series):
        try:
            return series == float(code)
        except (TypeError, ValueError):
            return pd.Series(False, index=series.index)
    text = series.astype("string").str.strip()
    return (text == str(code)).fillna(False)


def _apply_missing_codes(study: Study, processed: Processed) -> None:
    data = processed.data
    total = 0
    touched = []
    for name, variable in study.variables.items():
        if name not in data:
            continue
        counts = {}
        for code in (*study.missing_codes, *variable.missing_codes):
            hits = _code_matches(data[name], code)
            if hits.any():
                counts[str(code)] = int(hits.sum())
                data[name] = data[name].mask(hits)
        if counts:
            processed.missing_codes[name] = counts
            total += sum(counts.values())
            touched.append(name)
    if study.missing_codes or any(variable.missing_codes for variable in study.variables.values()):
        codes = ", ".join(map(str, study.missing_codes)) or "per-column codes"
        processed.records.append(
            Record(
                number=len(processed.records) + 1,
                kind="missing-codes",
                description=f"Read {codes} as missing",
                reason="Missing-value codes listed in study.yml.",
                rows_before=len(data),
                rows_after=len(data),
                values_changed=total,
                columns=tuple(touched),
            )
        )


def _apply_types(study: Study, processed: Processed) -> None:
    """Convert each described column to its declared type, counting values that don't fit."""
    data = processed.data
    changed = 0
    touched = []
    for name, variable in study.variables.items():
        if name not in data:
            continue
        column = data[name]
        if variable.type == "numeric" and not pd.api.types.is_numeric_dtype(column):
            converted = pd.to_numeric(column, errors="coerce")
            lost = column.notna() & converted.isna()
            if lost.any():
                examples = tuple(dict.fromkeys(map(str, column[lost])))[:5]
                processed.unreadable[name] = (int(lost.sum()), examples)
                changed += int(lost.sum())
                touched.append(name)
            data[name] = converted
        elif variable.type == "date" and not pd.api.types.is_datetime64_any_dtype(column):
            converted = pd.to_datetime(column, errors="coerce")
            lost = column.notna() & converted.isna()
            if lost.any():
                processed.unreadable[name] = (
                    int(lost.sum()),
                    tuple(dict.fromkeys(map(str, column[lost])))[:5],
                )
                changed += int(lost.sum())
                touched.append(name)
            data[name] = converted
        elif variable.type in ("categorical", "binary", "text"):
            data[name] = _as_labels(column)
    if changed:
        processed.records.append(
            Record(
                number=len(processed.records) + 1,
                kind="types",
                description="Set values that could not be read as their declared type to missing",
                reason="Types declared under `variables:` in study.yml; the audit lists each value.",
                rows_before=len(data),
                rows_after=len(data),
                values_changed=changed,
                columns=tuple(touched),
            )
        )


def _as_labels(column: pd.Series) -> pd.Series:
    """Category labels as text, writing whole-number codes as "2" rather than "2.0"."""
    if pd.api.types.is_float_dtype(column):
        whole = column.dropna()
        if len(whole) and (whole == whole.round()).all():
            return column.astype("Int64").astype("string")
    return column.astype("string").str.strip()


def _condition_mask(study: Study, condition: Condition, data: pd.DataFrame) -> pd.Series:
    column = data[condition.column]
    if condition.test == "missing":
        return column.isna() if condition.value in (None, True) else column.notna()
    if condition.test in ("in", "not_in", "equals", "not_equals"):
        values = condition.value if condition.test in ("in", "not_in") else [condition.value]
        hits = pd.Series(False, index=data.index)
        for value in values:
            hits |= _code_matches(column, value)
        return hits if condition.test in ("in", "equals") else ~hits & column.notna()
    numbers = pd.to_numeric(column, errors="coerce")
    limit = float(condition.value)
    compare = {
        "below": numbers < limit,
        "above": numbers > limit,
        "at_most": numbers <= limit,
        "at_least": numbers >= limit,
    }[condition.test]
    return compare.fillna(False)


def _matches(study: Study, step: Step, data: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=data.index)
    for condition in step.where:
        mask &= _condition_mask(study, condition, data)
    if step.ids:
        id_values = data[step.id_column].astype("string").str.strip()
        mask &= id_values.isin([str(value) for value in step.ids]).fillna(False)
    return mask.astype(bool)


def _transform(step: Step, data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    data = data.copy()
    values = pd.to_numeric(data[step.column], errors="coerce").astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = {
            "log": lambda: np.log(values.where(values > 0)),
            "log10": lambda: np.log10(values.where(values > 0)),
            "log2": lambda: np.log2(values.where(values > 0)),
            "sqrt": lambda: np.sqrt(values.where(values >= 0)),
            "square": lambda: values**2,
            "reciprocal": lambda: 1 / values.where(values != 0),
            "zscore": lambda: (values - values.mean()) / values.std(ddof=1),
            "center": lambda: values - values.mean(),
            "scale": lambda: values * float(step.factor),
        }[step.apply]()
    data[step.to or step.column] = result
    return data, int(result.notna().sum())


def _recode(step: Step, data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    data = data.copy()
    labels = _as_labels(data[step.column])
    mapping = {str(old): new for old, new in step.mapping.items()}
    recoded = labels.map(lambda value: mapping.get(value, value) if pd.notna(value) else value)
    changed = int((labels.isin(list(mapping))).fillna(False).sum())
    data[step.to or step.column] = recoded.astype("string")
    return data, changed


def _order_levels(study: Study, processed: Processed) -> None:
    """Make categorical columns ordered as their ``levels`` say, others in order of appearance."""
    for name, variable in study.variables.items():
        if variable.type not in ("categorical", "binary") or name not in processed.data:
            continue
        column = processed.data[name].astype("string")
        present = list(dict.fromkeys(column.dropna()))
        declared = [str(level) for level in variable.levels]
        order = declared + [level for level in present if level not in declared]
        processed.data[name] = pd.Categorical(column, categories=order, ordered=bool(declared))


def levels_of(data: pd.DataFrame, column: str) -> list[str]:
    """The levels of a categorical column that occur in ``data``, in study order."""
    series = data[column]
    if isinstance(series.dtype, pd.CategoricalDtype):
        present = set(series.dropna())
        return [level for level in series.cat.categories if level in present]
    return list(dict.fromkeys(series.dropna().astype(str)))
