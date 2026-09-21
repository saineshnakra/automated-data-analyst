"""Optional evidence-grounded narrative synthesis through the Responses API."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from pydantic import BaseModel, Field

from business_insights import BusinessBrief
from metrics import resolve_metric
from nlq import (
    AGGREGATION_LABELS,
    QueryAnswer,
    QueryPlan,
    ValueFilter,
    execute_plan,
    plan_accounts_for_numbers,
    question_is_representable,
)
from schema import ColumnRoles, looks_like_identifier


class AIAction(BaseModel):
    title: str = Field(max_length=90)
    recommendation: str = Field(max_length=320)
    evidence: str = Field(max_length=320)
    confidence: Literal["high", "medium", "low"]


class AINarrative(BaseModel):
    executive_summary: str = Field(max_length=650)
    strategic_read: str = Field(max_length=900)
    actions: list[AIAction] = Field(min_length=1, max_length=3)
    watchouts: list[str] = Field(max_length=3)


@dataclass(frozen=True)
class AIConfig:
    model: str
    reasoning_effort: Literal["none", "low", "medium", "high"]
    label: str


MODEL_PRESETS = {
    "Fast · Luna": AIConfig("gpt-5.6-luna", "low", "Fast · Luna"),
    "Deep · Terra": AIConfig("gpt-5.6-terra", "medium", "Deep · Terra"),
}

DEFAULT_PRESET = "Fast · Luna"


class _Responses(Protocol):
    def parse(self, **kwargs: object) -> object: ...


class _Client(Protocol):
    responses: _Responses


SYSTEM_INSTRUCTIONS = """You are ADA's strategic interpretation layer.
Use only the supplied deterministic calculations and business context.
Never invent numbers, entities, benchmarks, causes, or certainty.
Distinguish observed evidence from hypotheses. Make actions specific, testable, and prioritized.
If evidence is insufficient, state the limitation instead of filling the gap.
Write for an operator who needs the decision, not an analytics lecture."""


def build_ai_payload(brief: BusinessBrief, *, context: str = "") -> str:
    """Serialize only computed evidence; raw uploaded rows never enter the prompt."""
    payload = {
        "business_context": context.strip() or "Not provided",
        "detected_schema": asdict(brief.roles),
        "executive_headline": brief.headline,
        "computed_summary": brief.summary,
        "computed_evidence": [asdict(item) for item in brief.evidence],
        "deterministic_recommendations": [asdict(item) for item in brief.recommendations],
        "task": (
            "Synthesize the business meaning, choose up to three decision-ready actions, and list "
            "material watchouts. Anchor every action to supplied evidence."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def generate_ai_narrative(
    brief: BusinessBrief,
    *,
    api_key: str,
    config: AIConfig,
    context: str = "",
    safety_identifier: str,
    client: _Client | None = None,
) -> AINarrative:
    """Generate a typed narrative while keeping the deterministic brief authoritative."""
    if not api_key.strip():
        raise ValueError("An API key is required for the optional AI narrative.")
    if client is None:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=25.0, max_retries=1)

    response = client.responses.parse(
        model=config.model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=build_ai_payload(brief, context=context),
        text_format=AINarrative,
        reasoning={"effort": config.reasoning_effort},
        max_output_tokens=1_400,
        safety_identifier=safety_identifier,
        store=False,
    )
    narrative = getattr(response, "output_parsed", None)
    if narrative is None:
        raise RuntimeError("The strategy agent returned no structured narrative.")
    if isinstance(narrative, AINarrative):
        return narrative
    return AINarrative.model_validate(narrative)


class AIQueryFilter(BaseModel):
    column: str
    value: str = Field(max_length=120)


class AIQueryPlan(BaseModel):
    """Typed plan the model must emit; execution always happens locally."""

    answerable: bool
    intent: Literal["aggregate", "count", "rank", "breakdown", "trend", "growth"] = "aggregate"
    aggregation: Literal["sum", "mean", "median", "min", "max", "count"] = "sum"
    measure: str | None = None
    dimension: str | None = None
    top_n: int | None = Field(default=None, ge=1, le=50)
    ascending: bool = False
    filters: list[AIQueryFilter] = Field(default_factory=list, max_length=4)
    year: int | None = Field(default=None, ge=1800, le=2299)
    month: int | None = Field(default=None, ge=1, le=12)
    # A single day ("2024-01-31", "January 15") or a calendar quarter ("Q1").
    day: int | None = Field(default=None, ge=1, le=31)
    quarter: int | None = Field(default=None, ge=1, le=4)
    grain: Literal["D", "W", "M", "Q", "Y"] | None = None
    # Distinct entities rather than rows: "how many customers" counts this column's unique values.
    count_column: str | None = None


PLANNER_CONFIG = AIConfig("gpt-5.6-luna", "low", "Query planner")

PLANNER_INSTRUCTIONS = """You translate one business question about a single table into a strict query plan.
Use only the listed column names, exactly as written; never invent a column.
Filter values may only be phrases quoted from the question itself.
For "how many <entities>", set intent="count" and count_column to the column that identifies
one entity; leave it unset to count rows.
A single date sets year, month and day; a quarter sets quarter (1-4) and never month.
Every number in the question must appear in the plan (year, month, day, quarter, top_n or a
filter value); if one cannot, the question is not answerable.
If the schema cannot answer the question, set answerable to false instead of guessing."""


def build_query_schema(dataframe: pd.DataFrame, roles: ColumnRoles) -> list[dict[str, str]]:
    """Describe columns for the planner without exposing a single cell value."""
    schema: list[dict[str, str]] = []
    for column in dataframe.columns:
        series = dataframe[column]
        if is_datetime64_any_dtype(series):
            kind = "datetime"
        elif is_numeric_dtype(series):
            kind = "numeric"
        else:
            kind = "category"
        if column == roles.measure:
            role = "primary measure"
        elif column == roles.date:
            role = "date"
        elif column == roles.dimension:
            role = "primary dimension"
        elif column == roles.identifier:
            role = "identifier"
        elif column in roles.dimensions:
            role = "dimension"
        else:
            role = "other"
        schema.append({"column": column, "type": kind, "role": role})
    return schema


def build_planner_payload(question: str, dataframe: pd.DataFrame, roles: ColumnRoles) -> str:
    payload = {
        "question": question.strip(),
        "columns": build_query_schema(dataframe, roles),
        "task": "Emit the single best query plan for this question, or set answerable to false.",
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _resolve_filter(item: AIQueryFilter, dataframe: pd.DataFrame) -> ValueFilter | None:
    """Map a quoted filter phrase onto real values; refuse rather than guess."""
    if item.column not in dataframe.columns:
        return None
    wanted = item.value.casefold().strip()
    matched = tuple(
        str(value)
        for value in dataframe[item.column].dropna().unique()
        if str(value).casefold().strip() == wanted
    )
    return ValueFilter(column=item.column, values=matched) if matched else None


# Intents that group, and so cannot run without a dimension to group by.
GROUPING_INTENTS = ("rank", "breakdown")
# Intents whose executor reads plan.dimension at all. Growth ranks segments by
# their change, so it uses one; a plain aggregate never does.
DIMENSION_INTENTS = ("rank", "breakdown", "growth")
# Intents that resample onto a period grain.
GRAIN_INTENTS = ("trend", "growth")
# Intents whose executor works on period sums and ignores plan.aggregation.
PERIOD_INTENTS = ("trend", "growth")
# Above this share of distinct values a column identifies rows rather than
# grouping them, and a "breakdown" by it is one row per record.
IDENTIFIER_UNIQUENESS = 0.9


def _usable_measure(dataframe: pd.DataFrame, column: str, aggregation: str) -> bool:
    """Counting works on anything; arithmetic does not."""
    if aggregation == "count":
        return True
    return is_numeric_dtype(dataframe[column])


# Below this many rows, every value being distinct says nothing: a two-row
# regional summary has two distinct regions and is exactly what a breakdown
# is for.
IDENTIFIER_EVIDENCE_ROWS = 25


def _usable_dimension(dataframe: pd.DataFrame, column: str) -> bool:
    """A segment groups records together. A timestamp or an id does not.

    Uniqueness on its own is not evidence of an identifier -- a pre-aggregated
    table is unique by construction. So a column is refused when its name
    says it is a key, when it is a timestamp, when it is a continuous number,
    or when it is unique across enough rows for that to mean something.
    """
    series = dataframe[column]
    if is_datetime64_any_dtype(series):
        return False
    present = int(series.notna().sum())
    if not present:
        return False
    if looks_like_identifier(column, series):
        return False
    distinct = int(series.nunique(dropna=True))
    if is_numeric_dtype(series) and not is_bool_dtype(series):
        # A continuous measure is not a segment, and grouping by it produces a
        # row per distinct value.
        if (series.dropna() % 1 != 0).any() or distinct > present * IDENTIFIER_UNIQUENESS:
            return False
    # Uniqueness alone does not make a text column a key: a pre-aggregated
    # table with thirty labelled rows is unique by construction. Uniqueness
    # across enough rows AND values shaped like codes -- "T-0042", "INV0007"
    # -- does.
    if present >= IDENTIFIER_EVIDENCE_ROWS and distinct > present * IDENTIFIER_UNIQUENESS:
        # Code-shaped values ("T-0042", "INV0007") are a key beyond argument.
        # But a column that is unique across enough rows is not a segment
        # whatever its values look like: grouping sixty distinct email
        # addresses gives sixty groups of one, which is the table again with a
        # chart drawn on it. The code test decides how confident the refusal
        # is, not whether to refuse - requiring it let every unique free-text
        # column through, which is the failure this guard exists to stop.
        return False
    return True


_CODE_SHAPE = re.compile(r"^[A-Za-z]{0,6}[-_ ]?\d{2,}[A-Za-z0-9-]*$")


def _values_look_like_codes(series: pd.Series, sample: int = 200) -> bool:
    values = series.dropna().astype(str).head(sample)
    if values.empty:
        return False
    return bool(values.str.match(_CODE_SHAPE).mean() >= 0.8)


def _to_query_plan(
    parsed: AIQueryPlan, dataframe: pd.DataFrame, roles: ColumnRoles
) -> QueryPlan | None:
    """Turn a model's proposal into a plan ADA will execute exactly as described.

    Everything the executor would ignore is refused or stripped here rather
    than shown to the user for approval. An approval gate that describes a
    calculation which is not the one that runs is worse than no gate at all,
    because it buys confidence without earning it.
    """
    measure = parsed.measure or roles.measure
    dimension = parsed.dimension
    for column in (parsed.measure, parsed.dimension, parsed.count_column):
        if column is not None and column not in dataframe.columns:
            return None
    if parsed.count_column is not None and parsed.intent != "count":
        return None
    if parsed.intent in PERIOD_INTENTS and not roles.date:
        return None
    # The period executors combine each period the way the measure combines:
    # amounts are summed, rates are averaged. A plan asking for the other
    # operation would be approved with one sentence and executed with another,
    # so it is refused rather than silently corrected.
    if parsed.intent in PERIOD_INTENTS and parsed.aggregation != resolve_metric(measure).aggregation:
        return None
    if parsed.intent == "aggregate" and not measure:
        return None
    if measure is not None and not _usable_measure(dataframe, measure, parsed.aggregation):
        return None
    if parsed.intent in GROUPING_INTENTS:
        dimension = dimension or roles.dimension
        if not dimension:
            return None
    if dimension is not None:
        if dimension == measure or not _usable_dimension(dataframe, dimension):
            return None
    # A time filter needs a column to apply it to; without one the executor
    # quietly returns the all-time figure under a scoped-looking sentence.
    scoped_in_time = any(part is not None for part in (parsed.year, parsed.month, parsed.day, parsed.quarter))
    if scoped_in_time and not roles.date:
        return None
    # A day without a month, or a quarter alongside a month, is not one scope.
    if parsed.day is not None and parsed.month is None:
        return None
    if parsed.quarter is not None and parsed.month is not None:
        return None
    filters: list[ValueFilter] = []
    for item in parsed.filters:
        resolved = _resolve_filter(item, dataframe)
        if resolved is None:
            return None
        filters.append(resolved)
    return QueryPlan(
        intent=parsed.intent,
        aggregation=parsed.aggregation,
        measure=measure if parsed.intent != "count" else None,
        count_column=parsed.count_column if parsed.intent == "count" else None,
        # Every modifier the chosen intent's executor would ignore is dropped
        # here, so the approval sentence cannot advertise one.
        dimension=dimension if parsed.intent in DIMENSION_INTENTS else None,
        top_n=parsed.top_n if parsed.intent == "rank" else None,
        ascending=parsed.ascending,
        filters=tuple(filters),
        year=parsed.year,
        month=parsed.month,
        day=parsed.day,
        quarter=parsed.quarter,
        grain=parsed.grain if parsed.intent in GRAIN_INTENTS else None,
        source="ai",
    )


def plan_query_with_ai(
    question: str,
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    *,
    api_key: str,
    safety_identifier: str,
    config: AIConfig = PLANNER_CONFIG,
    client: _Client | None = None,
) -> QueryPlan | None:
    """Ask the model for a typed plan over the schema; execution stays local."""
    if not api_key.strip():
        raise ValueError("An API key is required for the optional AI query planner.")
    if not question.strip():
        return None
    # The planner emits the same plan shape the rules do, so a question no
    # plan can hold -- two metrics, an arithmetic expression, two months, a
    # comparison operator -- is refused here as well. Asking the model would
    # only get back the nearest question the plan can hold, presented for
    # approval as though it were the one asked.
    if not question_is_representable(question, dataframe, roles):
        return None
    if client is None:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=25.0, max_retries=1)

    response = client.responses.parse(
        model=config.model,
        instructions=PLANNER_INSTRUCTIONS,
        input=build_planner_payload(question, dataframe, roles),
        text_format=AIQueryPlan,
        reasoning={"effort": config.reasoning_effort},
        max_output_tokens=500,
        safety_identifier=safety_identifier,
        store=False,
    )
    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        return None
    if not isinstance(parsed, AIQueryPlan):
        parsed = AIQueryPlan.model_validate(parsed)
    if not parsed.answerable:
        return None
    plan = _to_query_plan(parsed, dataframe, roles)
    if plan is not None and not plan_accounts_for_numbers(question, plan):
        # "Revenue 2024-01-31" planned as January 2024 dropped the 31, and
        # with it the question.
        return None
    return plan


MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}
GRAIN_LABELS = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}


def _scope_clause(plan: QueryPlan) -> str:
    parts = [
        f"{item.column} = {', '.join(item.values)}" for item in plan.filters if item.values
    ]
    clause = f" for {' and '.join(parts)}" if parts else ""
    if plan.month is not None:
        label = MONTH_NAMES.get(plan.month, str(plan.month))
        if plan.day is not None:
            label = f"{label} {plan.day}"
        clause += f" {'on' if plan.day else 'in'} {label} {plan.year}" if plan.year is not None else (
            f" {'on' if plan.day else 'in'} {label}"
        )
    elif plan.quarter is not None:
        clause += f" in Q{plan.quarter} {plan.year}" if plan.year is not None else f" in Q{plan.quarter}"
    elif plan.year is not None:
        clause += f" in {plan.year}"
    return clause


def describe_query_plan(plan: QueryPlan) -> str:
    """Say, in a sentence, exactly what execute_plan will do with this plan.

    The sentence is the whole value of the approval step, so it is built per
    intent rather than from whichever fields happen to be set. A description
    that mentions a grouping or a ranking the executor ignores teaches the
    reader to trust a gate that is not checking anything.
    """
    measure = plan.measure or "records"
    aggregation = AGGREGATION_LABELS[plan.aggregation]
    grain = GRAIN_LABELS.get(plan.grain or "", "period")

    if plan.intent == "count":
        if plan.count_column:
            description = f"count the distinct {plan.count_column} values"
        else:
            description = "count the matching records"
    elif plan.intent == "trend":
        combined = resolve_metric(plan.measure).combines_as if plan.measure else "count"
        description = f"track {combined} {measure} per {grain} over time"
    elif plan.intent == "growth":
        combined = resolve_metric(plan.measure).combines_as if plan.measure else "count"
        if plan.dimension:
            description = (
                f"rank each {plan.dimension} by how much its {combined} {measure} changed "
                f"from the previous {grain} to the latest one"
            )
        else:
            description = (
                f"compare {combined} {measure} in the latest {grain} with the previous one"
            )
    elif plan.intent == "rank":
        direction = "lowest" if plan.ascending else "highest"
        showing = f"the {plan.top_n} {direction}" if plan.top_n else f"every one, {direction} first"
        described = "rows" if plan.aggregation == "count" else f"{aggregation.lower()} {measure}"
        description = f"rank {plan.dimension} by {described}, showing {showing}"
    elif plan.intent == "breakdown":
        # The executor honours every aggregation here, so the sentence must
        # name the one chosen. Hard-coding "total" once described a mean as a
        # sum, which is the exact lie the approval step exists to prevent.
        described = "rows" if plan.aggregation == "count" else f"{aggregation.lower()} {measure}"
        description = f"break {described} down by {plan.dimension}"
    else:
        description = f"{aggregation.lower()} of {measure}"

    description += _scope_clause(plan)
    return description[0].upper() + description[1:] + "."


def execute_approved_ai_plan(
    question: str,
    plan: QueryPlan,
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    *,
    approved: bool,
) -> QueryAnswer | None:
    """Execute a validated AI plan only after an explicit approval decision."""
    if not approved:
        return None
    executed = execute_plan(plan, dataframe, roles)
    return QueryAnswer(
        question=question,
        plan=executed.plan,
        answer=executed.answer,
        calculation=executed.calculation,
        table=executed.table,
        chart=executed.chart,
    )


def narrative_to_markdown(narrative: AINarrative, *, model: str) -> str:
    lines = [
        "## Optional AI strategic read",
        "",
        narrative.executive_summary,
        "",
        narrative.strategic_read,
        "",
        "### Recommended actions",
        "",
    ]
    for action in narrative.actions:
        lines.extend(
            [
                f"#### {action.title} · {action.confidence.title()} confidence",
                "",
                action.recommendation,
                "",
                f"_Evidence: {action.evidence}_",
                "",
            ]
        )
    if narrative.watchouts:
        lines.extend(["### Watchouts", ""])
        lines.extend(f"- {item}" for item in narrative.watchouts)
        lines.append("")
    lines.extend(
        [
            f"_Generated with {model} from the calculated evidence above; raw rows were not sent._",
            "",
        ]
    )
    return "\n".join(lines)
