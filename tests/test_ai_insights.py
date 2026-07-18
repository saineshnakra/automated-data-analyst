import json
import unittest
from types import SimpleNamespace

from ai_insights import (
    MODEL_PRESETS,
    AIAction,
    AINarrative,
    build_ai_payload,
    generate_ai_narrative,
    narrative_to_markdown,
)
from business_insights import analyze_business, detect_roles
from demo_data import make_demo_data


class FakeResponses:
    def __init__(self, narrative: AINarrative):
        self.narrative = narrative
        self.arguments = None

    def parse(self, **kwargs: object) -> object:
        self.arguments = kwargs
        return SimpleNamespace(output_parsed=self.narrative)


class FakeClient:
    def __init__(self, narrative: AINarrative):
        self.responses = FakeResponses(narrative)


class AIInsightTests(unittest.TestCase):
    def setUp(self):
        dataframe = make_demo_data(rows=240)
        self.brief = analyze_business(dataframe, detect_roles(dataframe))
        self.narrative = AINarrative(
            executive_summary="Growth is concentrated in a small set of operating signals.",
            strategic_read="Validate whether the latest movement persists before changing the plan.",
            actions=[
                AIAction(
                    title="Validate the leading segment",
                    recommendation="Compare the leader with the rest of the portfolio next period.",
                    evidence="The deterministic segment calculation identifies a clear leader.",
                    confidence="medium",
                )
            ],
            watchouts=["The file establishes correlation, not causation."],
        )

    def test_payload_contains_computed_evidence_not_raw_rows(self):
        payload = json.loads(build_ai_payload(self.brief, context="Weekly revenue review"))

        self.assertEqual(payload["business_context"], "Weekly revenue review")
        self.assertTrue(payload["computed_evidence"])
        self.assertTrue(payload["deterministic_recommendations"])
        self.assertNotIn("rows", payload)
        self.assertNotIn("records", payload)

    def test_generation_uses_typed_responses_contract(self):
        client = FakeClient(self.narrative)
        config = MODEL_PRESETS["Fast · Luna"]

        result = generate_ai_narrative(
            self.brief,
            api_key="test-key",
            config=config,
            context="Weekly revenue review",
            safety_identifier="anonymous-session",
            client=client,
        )

        self.assertEqual(result, self.narrative)
        arguments = client.responses.arguments
        self.assertIsNotNone(arguments)
        assert arguments is not None
        self.assertEqual(arguments["model"], "gpt-5.6-luna")
        self.assertEqual(arguments["reasoning"], {"effort": "low"})
        self.assertIs(arguments["text_format"], AINarrative)
        self.assertEqual(arguments["safety_identifier"], "anonymous-session")
        self.assertFalse(arguments["store"])

    def test_api_key_is_required_only_for_optional_narrative(self):
        with self.assertRaisesRegex(ValueError, "API key"):
            generate_ai_narrative(
                self.brief,
                api_key=" ",
                config=MODEL_PRESETS["Fast · Luna"],
                safety_identifier="anonymous-session",
            )

    def test_markdown_preserves_ai_provenance(self):
        report = narrative_to_markdown(self.narrative, model="gpt-5.6-luna")

        self.assertIn("Optional AI strategic read", report)
        self.assertIn("Validate the leading segment", report)
        self.assertIn("gpt-5.6-luna", report)
        self.assertIn("raw rows were not sent", report)


if __name__ == "__main__":
    unittest.main()
