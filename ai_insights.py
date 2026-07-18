"""Optional evidence-grounded narrative synthesis through the Responses API."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from business_insights import BusinessBrief


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
