"""Optional evidence-grounded narrative synthesis through the Responses API."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from pydantic import BaseModel, Field

from business_insights import BusinessBrief, ColumnRoles
from nlq import QueryPlan, ValueFilter


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
    year: int | None = Field(default=None, ge=1900, le=2100)
    month: int | None = Field(default=None, ge=1, le=12)
    grain: Literal["D", "W", "M", "Q", "Y"] | None = None


PLANNER_CONFIG = AIConfig("gpt-5.6-luna", "low", "Query planner")

PLANNER_INSTRUCTIONS = """You translate one business question about a single table into a strict query plan.
Use only the listed column names, exactly as written; never invent a column.
Filter values may only be phrases quoted from the question itself.
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


def _to_query_plan(
    parsed: AIQueryPlan, dataframe: pd.DataFrame, roles: ColumnRoles
) -> QueryPlan | None:
    measure = parsed.measure or roles.measure
    dimension = parsed.dimension
    for column in (parsed.measure, parsed.dimension):
        if column is not None and column not in dataframe.columns:
            return None
    if parsed.intent in ("trend", "growth") and not roles.date:
        return None
    if parsed.intent == "aggregate" and not measure:
        return None
    if parsed.intent in ("rank", "breakdown"):
        dimension = dimension or roles.dimension
        if not dimension:
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
        measure=measure,
        dimension=dimension,
        top_n=parsed.top_n,
        ascending=parsed.ascending,
        filters=tuple(filters),
        year=parsed.year,
        month=parsed.month,
        grain=parsed.grain,
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
    return _to_query_plan(parsed, dataframe, roles)


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
