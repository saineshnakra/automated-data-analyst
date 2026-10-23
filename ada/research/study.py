"""The study description that ``ada build`` works from.

It lives in ``study.yml`` beside the data. ``ada init`` writes a first draft
from the data and the researcher's answers; after that it is the researcher's
file, and ADA reads it but never rewrites it. Everything a result depends on is
in it: what a row is, the design, each variable's role and unit, every
exclusion and its reason, which analyses were planned, and what may be shared.
ADA checks the whole file before building and stops on any problem, so a typo
cannot quietly change a result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Study designs ADA can analyse. Each sets how observations depend on each other.
DESIGNS = {
    "independent": "each row is a different subject",
    "paired": "each subject is measured under two conditions",
    "repeated": "each subject is measured several times or under several conditions",
    "clustered": "subjects are grouped in clusters, such as cells within animals or patients within sites",
}
ROLES = ("outcome", "group", "time", "covariate", "identifier", "other")
TYPES = ("numeric", "categorical", "binary", "date", "text")
MISSING_DATA = ("complete-case",)
OUTLIERS = ("keep", "flag")
ADJUSTMENTS = ("holm", "bonferroni", "fdr_bh", "none")
SHARING = ("processed", "raw", "demo", "none")
TABLE_FORMATS = ("csv", "docx", "tex", "md")
FIGURE_FORMATS = ("pdf", "svg", "png", "tiff", "eps")
FIGURE_TYPES = ("groups", "paired", "longitudinal", "scatter", "missingness", "diagnostics", "panel")
TESTS = (
    "auto",
    "welch-t",
    "student-t",
    "anova",
    "welch-anova",
    "mann-whitney",
    "kruskal",
    "paired-t",
    "wilcoxon",
    "mixed-model",
    "ols",
    "pearson",
    "spearman",
    "rates",
    "logistic",
)
#: How a change analysis treats time: one level per time point, or a straight line.
TIME_AS = ("categorical", "linear")
CONDITIONS = ("below", "above", "at_most", "at_least", "equals", "not_equals", "in", "not_in", "missing")
TRANSFORMS = ("log", "log10", "log2", "sqrt", "square", "reciprocal", "zscore", "center", "scale")


class StudyError(ValueError):
    """``study.yml`` is missing, unreadable, or describes something ADA can't build."""

    def __init__(self, problems: list[str], path: Path | None = None):
        self.problems = problems
        where = f"{path}: " if path else ""
        listed = "\n".join(f"  - {problem}" for problem in problems)
        super().__init__(f"{where}{len(problems)} problem{'s' if len(problems) != 1 else ''}:\n{listed}")


@dataclass(frozen=True)
class Author:
    name: str
    orcid: str = ""
    affiliation: str = ""


@dataclass(frozen=True)
class Variable:
    name: str
    role: str = "other"
    type: str = "numeric"
    label: str = ""
    unit: str = ""
    description: str = ""
    #: For categorical and binary variables: the levels in display order. The
    #: first is the reference level that the others are compared with.
    levels: tuple[Any, ...] = ()
    #: For numeric variables: the plausible range, [low, high]. Values outside it
    #: are listed in the audit, not removed.
    range: tuple[float | None, float | None] | None = None
    #: Codes in this column that mean "missing", on top of the study-wide ones.
    missing_codes: tuple[Any, ...] = ()
    #: Decimal places for this variable in the tables; None chooses from its spread.
    decimals: int | None = None

    @property
    def display(self) -> str:
        """The label with its unit, for tables and axes: "Viability (%)"."""
        label = self.label or self.name
        return f"{label} ({self.unit})" if self.unit and f"({self.unit})" not in label else label


@dataclass(frozen=True)
class Condition:
    column: str
    test: str
    value: Any = None

    def describe(self) -> str:
        words = {
            "below": "is below",
            "above": "is above",
            "at_most": "is at most",
            "at_least": "is at least",
            "equals": "equals",
            "not_equals": "does not equal",
            "in": "is one of",
            "not_in": "is not one of",
        }
        if self.test == "missing":
            return f"{self.column} is missing"
        value = ", ".join(map(str, self.value)) if isinstance(self.value, list | tuple) else self.value
        return f"{self.column} {words[self.test]} {value}"


@dataclass(frozen=True)
class Step:
    """One processing step from ``processing``, applied in order."""

    kind: str  # exclude, transform or recode
    reason: str
    #: exclude: rows where every condition holds are removed.
    where: tuple[Condition, ...] = ()
    #: exclude: rows whose subject (or the named identifier column) has one of these values.
    ids: tuple[Any, ...] = ()
    id_column: str | None = None
    #: transform and recode: the column read and the column written (the same column by default).
    column: str | None = None
    to: str | None = None
    apply: str | None = None
    factor: float | None = None
    mapping: dict[Any, Any] = field(default_factory=dict)

    def describe(self) -> str:
        if self.kind == "exclude":
            parts = [condition.describe() for condition in self.where]
            if self.ids:
                parts.append(f"{self.id_column} is one of {', '.join(map(str, self.ids))}")
            return "Exclude rows where " + " and ".join(parts)
        target = self.to or self.column
        if self.kind == "recode":
            pairs = ", ".join(f"{old} → {new}" for old, new in self.mapping.items())
            return f"Recode {self.column}: {pairs}" + (f" (into {target})" if target != self.column else "")
        operation = f"{self.apply} × {self.factor:g}" if self.apply == "scale" else self.apply
        return f"{target} = {operation}({self.column})"


@dataclass(frozen=True)
class Decisions:
    missing_data: str = "complete-case"
    outliers: str = "keep"
    multiple_comparisons: str = "holm"
    alpha: float = 0.05
    confidence: float = 0.95


@dataclass(frozen=True)
class AnalysisSpec:
    id: str
    outcome: str
    #: Exactly one of compare, relate and change is set.
    compare: str | None = None
    relate: str | None = None
    change: str | None = None
    #: For change: a group whose trajectories are compared.
    by: str | None = None
    covariates: tuple[str, ...] = ()
    planned: bool = False
    question: str = ""
    family: str = ""
    test: str = "auto"
    #: For change: time as categorical (one level per time point) or linear; None chooses from the
    #: number of time points.
    time_as: str | None = None

    @property
    def kind(self) -> str:
        return "compare" if self.compare else "relate" if self.relate else "change"

    @property
    def predictor(self) -> str:
        return self.compare or self.relate or self.change or ""


@dataclass(frozen=True)
class FigureSpec:
    id: str
    type: str
    x: str | None = None
    y: str | None = None
    by: str | None = None
    analysis: str | None = None
    panels: tuple[str, ...] = ()
    title: str = ""
    caption: str = ""
    width_mm: float | None = None
    height_mm: float | None = None


@dataclass(frozen=True)
class Output:
    dir: str = "paper-assets"
    journal: str = "default"
    tables: tuple[str, ...] = ("csv", "docx", "tex")
    figures: tuple[str, ...] = ("pdf", "svg", "png")
    width_mm: float | None = None
    height_mm: float | None = None
    dpi: int | None = None
    font: str | None = None
    font_size: float | None = None


@dataclass(frozen=True)
class Sharing:
    data: str = "processed"
    #: Columns that never leave the private build, such as names or dates of birth.
    withhold: tuple[str, ...] = ()
    #: Whether files that list individual rows (figure source data, excluded rows) may be shared.
    row_level: bool = True


@dataclass(frozen=True)
class Custom:
    """A researcher's own Python file whose tables and figures join the assets."""

    script: str


@dataclass(frozen=True)
class Study:
    path: Path
    title: str
    data_path: str
    observation: str
    design: str
    variables: dict[str, Variable]
    authors: tuple[Author, ...] = ()
    description: str = ""
    keywords: tuple[str, ...] = ()
    license: str = ""
    code_license: str = ""
    data_sheet: str | None = None
    data_sha256: str = ""
    missing_codes: tuple[Any, ...] = ()
    subject: str | None = None
    cluster: str | None = None
    time: str | None = None
    design_notes: str = ""
    #: False while the design is ADA's guess from the data (``ada init --yes``) and the researcher
    #: has not checked it; the methods draft then says so rather than state the design as fact.
    design_confirmed: bool = True
    steps: tuple[Step, ...] = ()
    decisions: Decisions = Decisions()
    analyses: tuple[AnalysisSpec, ...] = ()
    figures: tuple[FigureSpec, ...] = ()
    descriptives_by: str | None = None
    descriptives: tuple[str, ...] = ()
    output: Output = Output()
    sharing: Sharing = Sharing()
    custom: tuple[Custom, ...] = ()
    #: The file as parsed, so each asset's record can hash the parts it depends on.
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def root(self) -> Path:
        """The folder holding ``study.yml``; every path in it is relative to this."""
        return self.path.parent

    @property
    def data_file(self) -> Path:
        return (self.root / self.data_path).resolve()

    @property
    def out_dir(self) -> Path:
        return (self.root / self.output.dir).resolve()

    def variable(self, name: str) -> Variable:
        return self.variables.get(name) or Variable(name=name)

    def label(self, name: str) -> str:
        return self.variable(name).display

    def with_role(self, role: str) -> list[str]:
        return [name for name, variable in self.variables.items() if variable.role == role]

    def section_hash(self, *keys: str) -> str:
        """A short hash of the named parts of the file, such as ``"processing"`` or ``"analyses.primary"``.

        Each asset records the hash of the parts it was built from, so a later
        build can say which assets a change to ``study.yml`` touched.
        """
        parts = {key: _lookup(self.raw, key) for key in sorted(keys)}
        text = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()[:16]


def _lookup(raw: dict, key: str) -> Any:
    """``raw["analyses"]`` for "analyses"; the entry with that id for "analyses.primary"."""
    head, _, rest = key.partition(".")
    value = raw.get(head)
    if not rest:
        return value
    if isinstance(value, dict):
        return value.get(rest)
    if isinstance(value, list):
        return next((item for item in value if isinstance(item, dict) and item.get("id") == rest), None)
    return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: str | Path = "study.yml") -> Study:
    """Read and check ``study.yml``. Raises :class:`StudyError` listing every problem found."""
    import yaml  # noqa: PLC0415

    path = Path(path).resolve()
    if path.is_dir():
        path = path / "study.yml"
    if not path.is_file():
        raise StudyError([f"No study file at {path}. Run `ada init DATA_FILE` to write one."])
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise StudyError([f"It is not valid YAML: {error}"], path) from error
    if not isinstance(raw, dict):
        raise StudyError(["It should hold sections such as `study:`, `data:` and `variables:`."], path)
    return parse(raw, path)


def parse(raw: dict[str, Any], path: Path) -> Study:
    """Turn the parsed YAML into a :class:`Study`, collecting every problem before raising."""
    reader = _Reader(raw)
    study_section = reader.section("study")
    data = reader.section("data")
    design = reader.section("design")
    decisions_section = reader.section("decisions")
    output_section = reader.section("output")
    sharing_section = reader.section("sharing")

    variables = {}
    for name, entry in (reader.mapping(raw, "variables") or {}).items():
        variables[str(name)] = reader.variable(str(name), entry)

    design_type = reader.choice(design, "type", DESIGNS, "design.type", default="independent")
    study = Study(
        path=path,
        title=reader.text(study_section, "title", "study.title", required=True),
        description=reader.text(study_section, "description", "study.description"),
        authors=tuple(reader.authors(study_section.get("authors"))),
        keywords=tuple(map(str, reader.items(study_section, "keywords", "study.keywords"))),
        license=reader.text(study_section, "license", "study.license"),
        code_license=reader.text(study_section, "code_license", "study.code_license"),
        data_path=reader.text(data, "path", "data.path", required=True),
        data_sheet=data.get("sheet"),
        data_sha256=reader.text(data, "sha256", "data.sha256"),
        observation=reader.text(data, "observation", "data.observation", required=True),
        missing_codes=tuple(reader.items(data, "missing_codes", "data.missing_codes")),
        design=design_type,
        subject=design.get("subject"),
        cluster=design.get("cluster"),
        time=design.get("time"),
        design_notes=reader.text(design, "notes", "design.notes"),
        design_confirmed=reader.flag(design, "confirmed", "design.confirmed", default=True),
        variables=variables,
        steps=tuple(reader.steps(raw.get("processing"), design.get("subject"))),
        decisions=Decisions(
            missing_data=reader.choice(
                decisions_section,
                "missing_data",
                MISSING_DATA,
                "decisions.missing_data",
                default="complete-case",
            ),
            outliers=reader.choice(
                decisions_section, "outliers", OUTLIERS, "decisions.outliers", default="keep"
            ),
            multiple_comparisons=reader.choice(
                decisions_section,
                "multiple_comparisons",
                ADJUSTMENTS,
                "decisions.multiple_comparisons",
                default="holm",
            ),
            alpha=reader.fraction(decisions_section, "alpha", "decisions.alpha", 0.05),
            confidence=reader.fraction(decisions_section, "confidence", "decisions.confidence", 0.95),
        ),
        analyses=tuple(reader.analyses(raw.get("analyses"))),
        figures=tuple(reader.figures(raw.get("figures"))),
        descriptives_by=(reader.section("descriptives").get("by")),
        descriptives=tuple(
            map(str, reader.items(reader.section("descriptives"), "variables", "descriptives"))
        ),
        output=Output(
            dir=reader.text(output_section, "dir", "output.dir") or "paper-assets",
            journal=reader.text(output_section, "journal", "output.journal") or "default",
            tables=tuple(reader.formats(output_section, "tables", TABLE_FORMATS, ("csv", "docx", "tex"))),
            figures=tuple(reader.formats(output_section, "figures", FIGURE_FORMATS, ("pdf", "svg", "png"))),
            width_mm=reader.number(output_section, "width_mm", "output.width_mm"),
            height_mm=reader.number(output_section, "height_mm", "output.height_mm"),
            dpi=reader.number(output_section, "dpi", "output.dpi"),
            font=output_section.get("font"),
            font_size=reader.number(output_section, "font_size", "output.font_size"),
        ),
        sharing=Sharing(
            data=reader.choice(sharing_section, "data", SHARING, "sharing.data", default="processed"),
            withhold=tuple(map(str, reader.items(sharing_section, "withhold", "sharing.withhold"))),
            row_level=bool(sharing_section.get("row_level", True)),
        ),
        custom=tuple(Custom(script=str(item)) for item in reader.items(raw, "custom", "custom")),
        raw=raw,
    )
    reader.check(study)
    if reader.problems:
        raise StudyError(reader.problems, path)
    return study


class _Reader:
    """Reads sections of the parsed YAML, noting each problem instead of stopping at the first."""

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self.problems: list[str] = []

    def section(self, key: str) -> dict:
        value = self.raw.get(key)
        if value is None:
            return {}
        if not isinstance(value, dict):
            self.problems.append(f"`{key}:` should be a section of `name: value` lines.")
            return {}
        return value

    def mapping(self, parent: dict, key: str) -> dict | None:
        value = parent.get(key)
        if value is not None and not isinstance(value, dict):
            self.problems.append(f"`{key}:` should be a section with one entry per column.")
            return None
        return value

    def text(self, parent: dict, key: str, where: str, required: bool = False) -> str:
        value = parent.get(key)
        if value is None or value == "":
            if required:
                self.problems.append(f"`{where}` is required.")
            return ""
        return str(value).strip()

    def items(self, parent: dict, key: str, where: str) -> list:
        value = parent.get(key)
        if value is None:
            return []
        if isinstance(value, list | tuple):
            return list(value)
        return [value]

    def number(self, parent: dict, key: str, where: str) -> float | None:
        value = parent.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            self.problems.append(f"`{where}` should be a number, not {value!r}.")
            return None
        return value

    def fraction(self, parent: dict, key: str, where: str, default: float) -> float:
        value = self.number(parent, key, where)
        if value is None:
            return default
        if not 0 < value < 1:
            self.problems.append(f"`{where}` should be between 0 and 1, not {value}.")
            return default
        return float(value)

    def flag(self, parent: dict, key: str, where: str, default: bool) -> bool:
        value = parent.get(key)
        if value is None:
            return default
        if not isinstance(value, bool):
            self.problems.append(f"`{where}` should be true or false, not {value!r}.")
            return default
        return value

    def choice(self, parent: dict, key: str, allowed, where: str, default: str) -> str:
        value = parent.get(key)
        if value is None:
            return default
        if str(value) not in allowed:
            self.problems.append(f"`{where}: {value}` is not one of: {', '.join(allowed)}.")
            return default
        return str(value)

    def formats(self, parent: dict, key: str, allowed, default) -> list[str]:
        values = [str(item).lower() for item in self.items(parent, key, f"output.{key}")] or list(default)
        for value in values:
            if value not in allowed:
                self.problems.append(f"`output.{key}`: {value} is not one of: {', '.join(allowed)}.")
        return [value for value in values if value in allowed]

    def authors(self, value) -> list[Author]:
        authors = []
        for item in value or []:
            if isinstance(item, str):
                authors.append(Author(name=item))
            elif isinstance(item, dict) and item.get("name"):
                authors.append(
                    Author(
                        name=str(item["name"]),
                        orcid=str(item.get("orcid") or ""),
                        affiliation=str(item.get("affiliation") or ""),
                    )
                )
            else:
                self.problems.append(f"`study.authors`: each author needs a `name`, not {item!r}.")
        return authors

    def variable(self, name: str, entry) -> Variable:
        where = f"variables.{name}"
        if entry is None:
            entry = {}
        if not isinstance(entry, dict):
            self.problems.append(f"`{where}` should be a section with `role:` and `type:`.")
            return Variable(name=name)
        known = {
            "role",
            "type",
            "label",
            "unit",
            "description",
            "levels",
            "range",
            "missing_codes",
            "decimals",
        }
        for key in entry:
            if key not in known:
                self.problems.append(
                    f"`{where}.{key}` is not a setting ADA knows: {', '.join(sorted(known))}."
                )
        value_range = entry.get("range")
        if value_range is not None:
            if not (isinstance(value_range, list) and len(value_range) == 2):
                self.problems.append(f"`{where}.range` should be [low, high]; use null for an open end.")
                value_range = None
            else:
                value_range = tuple(value_range)
        decimals = entry.get("decimals")
        if decimals is not None and not (isinstance(decimals, int) and 0 <= decimals <= 6):
            self.problems.append(f"`{where}.decimals` should be a whole number from 0 to 6.")
            decimals = None
        return Variable(
            name=name,
            role=self.choice(entry, "role", ROLES, f"{where}.role", default="other"),
            type=self.choice(entry, "type", TYPES, f"{where}.type", default="numeric"),
            label=str(entry.get("label") or ""),
            unit=str(entry.get("unit") or ""),
            description=str(entry.get("description") or ""),
            levels=tuple(self.items(entry, "levels", f"{where}.levels")),
            range=value_range,
            missing_codes=tuple(self.items(entry, "missing_codes", f"{where}.missing_codes")),
            decimals=decimals,
        )

    def steps(self, value, subject: str | None) -> list[Step]:
        if value is None:
            return []
        if not isinstance(value, list):
            self.problems.append(
                "`processing:` should be a list of steps, each starting with `- exclude:`, "
                "`- transform:` or `- recode:`."
            )
            return []
        steps = []
        for number, item in enumerate(value, 1):
            where = f"processing step {number}"
            kinds = [
                key for key in ("exclude", "transform", "recode") if isinstance(item, dict) and key in item
            ]
            if len(kinds) != 1:
                self.problems.append(
                    f"{where} should have exactly one of `exclude:`, `transform:` or `recode:`."
                )
                continue
            kind = kinds[0]
            body = item[kind] or {}
            reason = str(item.get("reason") or body.get("reason") or "").strip()
            if not reason:
                self.problems.append(
                    f"{where} ({kind}) needs a `reason:`. Every exclusion and change is reported "
                    "with its reason."
                )
            if kind == "exclude":
                conditions = []
                where_value = body.get("where")
                for entry in (
                    where_value if isinstance(where_value, list) else [where_value] if where_value else []
                ):
                    condition = self.condition(entry, where)
                    if condition:
                        conditions.append(condition)
                ids = tuple(self.items(body, "ids", where))
                id_column = body.get("id_column") or subject
                if ids and not id_column:
                    self.problems.append(
                        f"{where} lists `ids` but names no `id_column` and the design has no `subject`."
                    )
                if not conditions and not ids:
                    self.problems.append(f"{where} needs `where:` conditions or `ids:` to exclude.")
                steps.append(
                    Step(kind="exclude", reason=reason, where=tuple(conditions), ids=ids, id_column=id_column)
                )
            elif kind == "transform":
                operation = body.get("apply")
                if operation not in TRANSFORMS:
                    self.problems.append(
                        f"{where}: `apply: {operation}` is not one of: {', '.join(TRANSFORMS)}."
                    )
                factor = body.get("factor")
                if operation == "scale" and not isinstance(factor, int | float):
                    self.problems.append(f"{where}: `apply: scale` needs a numeric `factor:`.")
                if not body.get("column"):
                    self.problems.append(f"{where} needs the `column:` to transform.")
                steps.append(
                    Step(
                        kind="transform",
                        reason=reason,
                        column=body.get("column"),
                        to=body.get("to"),
                        apply=operation,
                        factor=factor,
                    )
                )
            else:
                mapping = body.get("map")
                if not isinstance(mapping, dict) or not mapping:
                    self.problems.append(f"{where} needs a `map:` of old value: new value.")
                    mapping = {}
                if not body.get("column"):
                    self.problems.append(f"{where} needs the `column:` to recode.")
                steps.append(
                    Step(
                        kind="recode",
                        reason=reason,
                        column=body.get("column"),
                        to=body.get("to"),
                        mapping=dict(mapping),
                    )
                )
        return steps

    def condition(self, entry, where: str) -> Condition | None:
        if not isinstance(entry, dict) or "column" not in entry:
            self.problems.append(
                f"{where}: each `where` condition needs a `column:` and one of {', '.join(CONDITIONS)}."
            )
            return None
        tests = [key for key in entry if key != "column"]
        if len(tests) != 1 or tests[0] not in CONDITIONS:
            self.problems.append(
                f"{where}: the condition on {entry['column']} should have exactly one of "
                f"{', '.join(CONDITIONS)}."
            )
            return None
        test = tests[0]
        value = entry[test]
        if test in ("in", "not_in") and not isinstance(value, list):
            value = [value]
        return Condition(column=str(entry["column"]), test=test, value=value)

    def analyses(self, value) -> list[AnalysisSpec]:
        if value is None:
            return []
        if not isinstance(value, list):
            self.problems.append("`analyses:` should be a list, each entry starting with `- id:`.")
            return []
        found = []
        for number, item in enumerate(value, 1):
            if not isinstance(item, dict):
                self.problems.append(
                    f"analysis {number} should be a section with `id:`, `outcome:` and one of "
                    "`compare:`, `relate:` or `change:`."
                )
                continue
            ident = str(item.get("id") or f"analysis-{number}")
            where = f"analyses.{ident}"
            kinds = [key for key in ("compare", "relate", "change") if item.get(key)]
            if len(kinds) != 1:
                self.problems.append(
                    f"`{where}` needs exactly one of `compare:` (a group), `relate:` (a numeric "
                    "predictor) or `change:` (a time variable)."
                )
            if not item.get("outcome"):
                self.problems.append(f"`{where}` needs an `outcome:`.")
            if "planned" not in item:
                self.problems.append(
                    f"`{where}` needs `planned: true` (decided before seeing the data) or "
                    "`planned: false` (exploratory)."
                )
            found.append(
                AnalysisSpec(
                    id=ident,
                    outcome=str(item.get("outcome") or ""),
                    compare=item.get("compare"),
                    relate=item.get("relate"),
                    change=item.get("change"),
                    by=item.get("by"),
                    covariates=tuple(map(str, self.items(item, "covariates", where))),
                    planned=bool(item.get("planned", False)),
                    question=str(item.get("question") or ""),
                    family=str(item.get("family") or ident),
                    test=self.choice(item, "test", TESTS, f"{where}.test", default="auto"),
                    time_as=self.time_as(item, where),
                )
            )
        return found

    def time_as(self, item: dict, where: str) -> str | None:
        value = self.choice(item, "time_as", TIME_AS, f"{where}.time_as", default=None)
        if value and not item.get("change"):
            self.problems.append(f"`{where}.time_as` only applies to a `change:` analysis.")
            return None
        return value

    def figures(self, value) -> list[FigureSpec]:
        if value is None:
            return []
        if not isinstance(value, list):
            self.problems.append("`figures:` should be a list, each entry starting with `- id:`.")
            return []
        found = []
        for number, item in enumerate(value, 1):
            if not isinstance(item, dict):
                self.problems.append(f"figure {number} should be a section with `id:` and `type:`.")
                continue
            ident = str(item.get("id") or f"figure-{number}")
            found.append(
                FigureSpec(
                    id=ident,
                    type=self.choice(item, "type", FIGURE_TYPES, f"figures.{ident}.type", default="groups"),
                    x=item.get("x"),
                    y=item.get("y"),
                    by=item.get("by"),
                    analysis=item.get("analysis"),
                    panels=tuple(map(str, self.items(item, "panels", f"figures.{ident}.panels"))),
                    title=str(item.get("title") or ""),
                    caption=str(item.get("caption") or ""),
                    width_mm=self.number(item, "width_mm", f"figures.{ident}.width_mm"),
                    height_mm=self.number(item, "height_mm", f"figures.{ident}.height_mm"),
                )
            )
        return found

    def check(self, study: Study) -> None:
        """Cross-checks between sections: every named column is described, and the design is complete."""
        known = set(study.variables)

        def described(name: str | None, where: str) -> None:
            if name and name not in known:
                self.problems.append(f"`{where}` names {name!r}, which is not under `variables:`.")

        described(study.subject, "design.subject")
        described(study.cluster, "design.cluster")
        described(study.time, "design.time")
        described(study.descriptives_by, "descriptives.by")
        if study.design in ("paired", "repeated") and not study.subject:
            self.problems.append(
                f"A {study.design} design needs `design.subject`: the column naming who or what "
                "was measured more than once."
            )
        if study.design == "clustered" and not study.cluster:
            self.problems.append(
                "A clustered design needs `design.cluster`: the column naming the cluster each "
                "row belongs to."
            )
        for name, variable in study.variables.items():
            if variable.levels and variable.type not in ("categorical", "binary"):
                self.problems.append(
                    f"`variables.{name}.levels` only applies to categorical or binary variables."
                )
            if variable.range and variable.type != "numeric":
                self.problems.append(f"`variables.{name}.range` only applies to numeric variables.")
        ids = [analysis.id for analysis in study.analyses]
        for ident in {ident for ident in ids if ids.count(ident) > 1}:
            self.problems.append(f"Two analyses share the id {ident!r}.")
        for analysis in study.analyses:
            where = f"analyses.{analysis.id}"
            for key, name in (
                ("outcome", analysis.outcome),
                ("compare", analysis.compare),
                ("relate", analysis.relate),
                ("change", analysis.change),
                ("by", analysis.by),
            ):
                described(name, f"{where}.{key}")
            for name in analysis.covariates:
                described(name, f"{where}.covariates")
        figure_ids = {figure.id for figure in study.figures}
        for figure in study.figures:
            for key, name in (("x", figure.x), ("y", figure.y), ("by", figure.by)):
                described(name, f"figures.{figure.id}.{key}")
            if figure.analysis and figure.analysis not in ids:
                self.problems.append(
                    f"`figures.{figure.id}.analysis` names {figure.analysis!r}, which is not an analysis id."
                )
            for panel in figure.panels:
                if panel not in figure_ids:
                    self.problems.append(
                        f"`figures.{figure.id}.panels` names {panel!r}, which is not a figure id."
                    )
        for step in study.steps:
            for condition in step.where:
                described(condition.column, "processing")
            if step.kind != "exclude":
                described(step.column, "processing")
                if step.to and step.to not in known:
                    self.problems.append(
                        f"processing writes {step.to!r}; describe it under `variables:` too."
                    )
        for name in study.descriptives:
            described(name, "descriptives.variables")
