import tempfile
import unittest
from pathlib import Path

from research_helpers import assay_study

from ada.research.study import StudyError, load, parse


class ParseTests(unittest.TestCase):
    def parse(self, study: dict):
        return parse(study, Path("/tmp/study.yml"))

    def test_a_complete_study_parses(self):
        study = self.parse(assay_study())
        self.assertEqual(study.design, "independent")
        self.assertEqual(study.variable("Treatment").levels, ("Control", "Low", "High"))
        self.assertEqual(study.label("Viability (%)"), "Viability (%)")
        self.assertTrue(study.analyses[0].planned)
        self.assertEqual(study.analyses[0].kind, "compare")

    def test_every_problem_is_listed_at_once(self):
        broken = assay_study(design={"type": "crossover"})
        broken["variables"]["Treatment"]["role"] = "treatment"
        broken["analyses"] = [{"id": "a", "outcome": "Viability (%)"}]
        with self.assertRaises(StudyError) as raised:
            self.parse(broken)
        text = str(raised.exception)
        self.assertIn("design.type: crossover", text)
        self.assertIn("variables.Treatment.role", text)
        self.assertIn("exactly one of `compare:`", text)
        self.assertIn("planned: true", text)

    def test_an_exclusion_needs_a_reason(self):
        study = assay_study(processing=[{"exclude": {"where": {"column": "Viability (%)", "below": 0}}}])
        with self.assertRaises(StudyError) as raised:
            self.parse(study)
        self.assertIn("needs a `reason:`", str(raised.exception))

    def test_a_repeated_design_needs_a_subject(self):
        with self.assertRaises(StudyError) as raised:
            self.parse(assay_study(design={"type": "repeated"}))
        self.assertIn("design.subject", str(raised.exception))

    def test_names_must_be_described(self):
        study = assay_study(
            analyses=[{"id": "a", "outcome": "Viability", "compare": "Treatment", "planned": True}]
        )
        with self.assertRaises(StudyError) as raised:
            self.parse(study)
        self.assertIn("'Viability', which is not under `variables:`", str(raised.exception))

    def test_a_transform_target_must_be_described(self):
        study = assay_study(
            processing=[
                {"transform": {"column": "Passage", "to": "log_passage", "apply": "log"}, "reason": "Skewed."}
            ]
        )
        with self.assertRaises(StudyError) as raised:
            self.parse(study)
        self.assertIn("describe it under `variables:`", str(raised.exception))

    def test_section_hash_follows_only_the_named_parts(self):
        first = self.parse(assay_study())
        changed = assay_study()
        changed["study"]["title"] = "Another title"
        second = self.parse(changed)
        self.assertEqual(first.section_hash("analyses.primary"), second.section_hash("analyses.primary"))
        self.assertNotEqual(first.section_hash("study"), second.section_hash("study"))

    def test_load_explains_a_missing_file(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(StudyError) as raised:
            load(Path(folder) / "study.yml")
        self.assertIn("ada init", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
