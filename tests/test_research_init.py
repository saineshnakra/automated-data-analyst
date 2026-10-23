"""``ada init``: the design it proposes, and the analyses it writes only when the build would run them."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from research_helpers import assay

from ada.research import init, load
from ada.research.interview import guess


def recordings(people: int = 12, per_person: int = 8, seed: int = 7) -> pd.DataFrame:
    """Voice recordings of people numbered 1 to ``people`` in "subject#", several each (telemonitoring)."""
    rng = np.random.default_rng(seed)
    rows = []
    for person in range(1, people + 1):
        age, level = int(rng.integers(55, 80)), rng.normal(20, 5)
        for visit in range(per_person):
            rows.append(
                {
                    "subject#": person,
                    "age": age,
                    "sex": person % 2,
                    "test_time": round(visit * 7 + rng.uniform(0, 3), 2),
                    "motor_UPDRS": round(level + 0.1 * visit + rng.normal(0, 1), 3),
                    "Jitter(%)": round(abs(rng.normal(0.006, 0.002)), 5),
                }
            )
    return pd.DataFrame(rows)


def one_station(hours: int = 120, seed: int = 4) -> pd.DataFrame:
    """Hourly readings at a single station: every row has the same station name and site code."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "No": np.arange(1, hours + 1),
            "PM2.5": np.round(rng.gamma(2, 30, hours), 1),
            "TEMP": np.round(rng.normal(12, 8, hours), 1),
            "DEWP": np.round(rng.normal(0, 10, hours), 1),
            "station": "Aotizhongxin",
            "site": 3,
        }
    )


def write_and_init(frame: pd.DataFrame, root: Path, name: str = "data.csv", **options) -> Path:
    frame.to_csv(root / name, index=False)
    options.setdefault("say", lambda _: None)
    return init(root / name, study_path=root / "study.yml", **options)


def answering(answers: dict[str, list[str]], asked: list[str]):
    """An ``ask`` that answers each question containing a key with that key's answers in turn, else Enter."""

    def ask(question: str) -> str:
        asked.append(question)
        for key, replies in answers.items():
            if key in question and replies:
                return replies.pop(0)
        return ""

    return ask


class DesignConfirmationTests(unittest.TestCase):
    def test_a_design_accepted_with_yes_is_marked_unconfirmed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_and_init(assay(), Path(folder), interactive=False)
            self.assertIn("confirmed: false", path.read_text())
            self.assertFalse(load(path).design_confirmed)

    def test_answering_the_design_question_confirms_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_and_init(assay(), Path(folder), ask=answering({}, []))
            self.assertTrue(load(path).design_confirmed)


class RepeatedSubjectTests(unittest.TestCase):
    def test_repeated_number_codes_are_proposed_as_the_subject(self):
        with tempfile.TemporaryDirectory() as folder:
            messages: list[str] = []
            path = write_and_init(recordings(), Path(folder), interactive=False, say=messages.append)
            study = load(path)
            self.assertEqual(study.design, "repeated")
            self.assertEqual(study.subject, "subject#")
            self.assertEqual(study.variable("subject#").role, "identifier")
            self.assertFalse(study.design_confirmed)
            self.assertIn("confirmed: false", path.read_text())
            self.assertTrue(any("from subject# (12 values over 96 rows, 8 rows each)" in m for m in messages))

    def test_a_key_with_a_different_value_on_every_row_is_not_a_subject(self):
        frame = recordings(people=40, per_person=1).rename(columns={"subject#": "Patient ID"})
        with tempfile.TemporaryDirectory() as folder:
            frame.to_csv(Path(folder) / "data.csv", index=False)
            found = guess(Path(folder) / "data.csv", Path(folder))
        self.assertIn("Patient ID", found.identifiers)
        self.assertEqual(found.design, "independent")
        self.assertIsNone(found.subject)

    def test_counts_ratings_years_and_visit_numbers_are_not_subjects(self):
        rng = np.random.default_rng(9)
        frame = pd.DataFrame(
            {
                "children": rng.integers(0, 5, 80),
                "rating": rng.integers(1, 6, 80),
                "year": rng.integers(2015, 2020, 80),
                "Visit Number": np.tile([1, 2, 3, 4], 20),
                "Number": rng.integers(1, 6, 80),
                "income": np.round(rng.normal(40, 10, 80), 2),
            }
        )
        with tempfile.TemporaryDirectory() as folder:
            frame.to_csv(Path(folder) / "survey.csv", index=False)
            found = guess(Path(folder) / "survey.csv", Path(folder))
        self.assertEqual(found.design, "independent")
        self.assertIsNone(found.subject)
        self.assertIsNone(found.cluster)
        for name in ("children", "rating", "year"):
            with self.subTest(name=name):
                self.assertNotIn(name, found.identifiers)

    def test_a_column_with_one_value_is_never_the_subject_or_cluster(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_and_init(one_station(), Path(folder), interactive=False)
            study = load(path)
        self.assertEqual(study.design, "independent")
        self.assertIsNone(study.subject)
        self.assertIsNone(study.cluster)
        self.assertNotEqual(study.variable("station").role, "identifier")

    def test_the_interview_asks_to_confirm_a_repeated_design(self):
        with tempfile.TemporaryDirectory() as folder:
            asked: list[str] = []
            said: list[str] = []
            path = write_and_init(recordings(), Path(folder), ask=answering({}, asked), say=said.append)
            study = load(path)
        self.assertEqual(study.design, "repeated")
        self.assertTrue(study.design_confirmed)
        explained = next(text for text in said if "subject# repeats" in text)
        self.assertIn("12 values over 96 rows, 8 rows each", explained)
        self.assertIn("Answer independent only if each row is a different subject", explained)
        design = next(question for question in asked if question.startswith("Design ("))
        self.assertIn("[repeated]", design)

    def test_the_researcher_can_answer_independent(self):
        with tempfile.TemporaryDirectory() as folder:
            ask = answering({"Design (": ["independent"]}, [])
            study = load(write_and_init(recordings(), Path(folder), ask=ask))
        self.assertEqual(study.design, "independent")
        self.assertTrue(study.design_confirmed)

    def test_a_key_with_a_few_repeats_is_left_independent_but_pointed_out(self):
        frame = recordings(people=40, per_person=1).rename(columns={"subject#": "Patient ID"})
        frame.loc[39, "Patient ID"] = 1
        with tempfile.TemporaryDirectory() as folder:
            said: list[str] = []
            study = load(write_and_init(frame, Path(folder), ask=answering({}, []), say=said.append))
        self.assertEqual(study.design, "independent")
        self.assertTrue(any("Patient ID repeats on 2 of 40 rows" in text for text in said))

    def test_a_subject_with_one_value_is_asked_for_again(self):
        frame = recordings()
        frame["station"] = "Clinic A"
        with tempfile.TemporaryDirectory() as folder:
            said: list[str] = []
            ask = answering({"Column naming the subject": ["station", "subject#"]}, [])
            study = load(write_and_init(frame, Path(folder), ask=ask, say=said.append))
        self.assertEqual(study.subject, "subject#")
        self.assertTrue(any("station has a single value" in text for text in said))


if __name__ == "__main__":
    unittest.main()
