import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import ada
from ada.api import Refusal
from ada.cli import main
from ada.decisions import UnknownDecision
from ada.joins import ask_folder, folder_choices


def people() -> pd.DataFrame:
    """Four players; two share a first name and two a last name, so neither name alone picks one."""
    return pd.DataFrame({
        "playerID": ["bondsba01", "bondsbo01", "griffke02", "griffke01"],
        "nameFirst": ["Barry", "Bobby", "Ken", "Ken"],
        "nameLast": ["Bonds", "Bonds", "Griffey", "Griffey"],
        "birthYear": [1964, 1946, 1969, 1950],
    })


def batting() -> pd.DataFrame:
    """Home runs by season. By hand: Barry Bonds 2001 is 73; the three Barry Bonds seasons add up to 164."""
    return pd.DataFrame({
        "playerID": ["bondsba01", "bondsba01", "bondsba01", "bondsbo01", "griffke02", "griffke02",
                     "griffke01", "bondsbo01"],
        "yearID": [2001, 2002, 2003, 1973, 2001, 2002, 1980, 1974],
        "HR": [73, 46, 45, 39, 22, 8, 13, 21],
    })


class JoinTestCase(unittest.TestCase):
    def folder(self, files: dict[str, pd.DataFrame]) -> Path:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        root = Path(holder.name)
        for name, frame in files.items():
            frame.to_csv(root / name, index=False)
        return root


class PersonNameTests(unittest.TestCase):
    def setUp(self):
        self.analysis = ada.analyze(batting().merge(people(), on="playerID"))

    def test_a_first_and_last_name_pick_one_person(self):
        answer = self.analysis.ask("total HR for Barry Bonds")
        self.assertAlmostEqual(answer.value, 164.0)
        self.assertIn("nameFirst = Barry", answer.calculation)
        self.assertIn("nameLast = Bonds", answer.calculation)

    def test_names_beyond_the_value_index_are_still_found(self):
        # 10,050 last names are more than the parser indexes as values, so only the name rule finds one.
        frame = pd.DataFrame({"nameFirst": ["Ann", "Bo", "Cy"] * 3350,
                              "nameLast": [f"Name{index}" for index in range(10050)],
                              "HR": [index % 7 for index in range(10050)]})
        answer = ada.analyze(frame, row_limit=20000).ask("total HR for Bo Name4000")
        self.assertAlmostEqual(answer.value, 4000 % 7)
        self.assertIsInstance(ada.analyze(frame, row_limit=20000).ask("total HR for Ann Name4000"), Refusal)

    def test_two_columns_of_first_names_name_no_one(self):
        firsts = ["Ann", "Bo", "Cy"] * 3350
        frame = pd.DataFrame({"nameFirst": firsts, "first_name": firsts,
                              "nameLast": [f"Name{index}" for index in range(10050)], "HR": [1] * 10050})
        self.assertIsInstance(ada.analyze(frame, row_limit=20000).ask("total HR for Bo Name4000"), Refusal)


class TallyVerbTests(unittest.TestCase):
    def setUp(self):
        self.frame = batting().merge(people().drop(columns="birthYear"), on="playerID")

    def test_the_words_after_a_tally_verb_name_the_column(self):
        check = ada.check_claim(self.frame, "Barry Bonds hit 73 home runs in 2001.")
        self.assertTrue(check.verified)
        self.assertEqual(check.answer.calculation[:7], "sum(HR)")
        self.assertIn('Read "home runs" as HR.', check.assumptions)

    def test_a_tally_verb_before_a_column_name(self):
        frame = pd.DataFrame({"forename": ["Lewis", "Lewis", "Max"],
                              "surname": ["Hamilton", "Hamilton", "Verstappen"],
                              "points": [25.0, 18.5, 25.0]})
        self.assertTrue(ada.check_claim(frame, "Lewis Hamilton scored 43.5 points.").verified)
        self.assertEqual(ada.check_claim(frame, "Lewis Hamilton scored 40 points.").reason, "failed_check")

    def test_an_average_after_a_tally_verb_stays_an_average(self):
        frame = pd.DataFrame({"team": ["A", "A", "B"], "goals": [1.0, 3.0, 2.0]})
        check = ada.check_claim(frame, "Team A scored 2 goals on average.")
        self.assertTrue(check.verified)
        self.assertTrue(check.answer.calculation.startswith("mean(goals)"))

    def test_grammar_words_are_not_initials(self):
        from ada.claims import _initials, _named

        self.assertTrue(_initials("total home runs", "HR"))
        self.assertFalse(_initials("sales in December", "ID"))
        self.assertFalse(_named("sales in December", "ID"))


class JoinTests(JoinTestCase):
    def setUp(self):
        self.root = self.folder({"Batting.csv": batting(), "People.csv": people()})

    def test_a_claim_that_needs_two_files_is_checked_on_them_joined(self):
        check = ada.check_claim(self.root, "Barry Bonds hit 73 home runs in 2001.")
        self.assertTrue(check.verified)
        self.assertEqual(check.source.name, "Batting.csv joined to People.csv")
        self.assertIn("Joined Batting.csv to People.csv on Batting.playerID = People.playerID",
                      check.assumptions[0])
        self.assertEqual(check.to_dict()["decisions"][0]["key"], "join_links")

    def test_the_second_files_dates_are_left_out(self):
        # birthYear would make "in 2001" ambiguous; the year stays the batting season's.
        check = ada.check_claim(self.root, "Barry Bonds hit 73 home runs in 2001.")
        self.assertIn("yearID", check.answer.calculation)
        self.assertNotIn("birthYear", check.answer.calculation)

    def test_no_joins_when_the_caller_says_so(self):
        check = ada.check_claim(self.root, "Barry Bonds hit 73 home runs in 2001.",
                                decisions={"join_links": "none"})
        self.assertEqual(check.reason, "missing_context")
        self.assertIn("Nor does the 0 joins ADA tried (join_links=none)", check.message)

    def test_a_key_that_repeats_on_the_other_side_is_not_joined(self):
        # Each playerID has two rows in salaries.csv, so joining on it would repeat batting rows.
        salaries = pd.DataFrame({"playerID": [*people()["playerID"]] * 2, "salary": [1.0, 2.0, 3.0, 4.0] * 2})
        root = self.folder({"Batting.csv": batting(), "Salaries.csv": salaries})
        check = ada.check_claim(root, "Total salary for bondsba01 was 3.")
        self.assertNotEqual(check.source.name, "Batting.csv joined to Salaries.csv")

    def test_rows_with_no_match_stay_with_empty_columns(self):
        batted = pd.concat([batting(), pd.DataFrame({"playerID": ["nobody01"], "yearID": [2001], "HR": [5]})])
        root = self.folder({"Batting.csv": batted, "People.csv": people()})
        # A key with no match makes the link one to confirm: not joined until the answers confirm it.
        self.assertIsInstance(ask_folder(root, "total HR for Barry Bonds in 2001"), Refusal)
        confirmed = {"link:Batting.playerID->People.playerID": "yes"}
        answer = ask_folder(root, "total HR for Barry Bonds in 2001", answers=confirmed)
        self.assertAlmostEqual(answer.value, 73.0)
        self.assertIn("1 row unmatched", answer.answer)


class AskFolderTests(JoinTestCase):
    def test_a_question_one_file_answers(self):
        root = self.folder({"Batting.csv": batting(), "People.csv": people()})
        answer = ask_folder(root, "total HR")
        self.assertAlmostEqual(answer.value, float(batting()["HR"].sum()))
        self.assertIn("Answered from Batting.csv, the only file", answer.answer)
        self.assertEqual(answer.source.name, "Batting.csv")

    def test_a_question_two_files_answer_differently_is_refused(self):
        other = batting().assign(HR=batting()["HR"] * 2)
        root = self.folder({"Batting.csv": batting(), "Batting2.csv": other})
        answer = ask_folder(root, "total HR")
        self.assertIsInstance(answer, Refusal)
        self.assertIn("more than one file", answer.message)

    def test_a_question_that_needs_a_join(self):
        root = self.folder({"Batting.csv": batting(), "People.csv": people()})
        answer = ask_folder(root, "total HR for Barry Bonds in 2001")
        self.assertAlmostEqual(answer.value, 73.0)
        self.assertEqual(answer.decisions[0].key, "join_links")
        self.assertIn("Joined Batting.csv to People.csv", answer.scope.readings[0])

    def test_an_empty_folder(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.assertIsInstance(ask_folder(holder.name, "total HR"), Refusal)


class IntakeTests(JoinTestCase):
    def emissions(self) -> pd.DataFrame:
        return pd.DataFrame({
            "country": ["Kenya", "Nigeria", "Chile", "Africa", "World"],
            "co2": [10.0, 20.0, 30.0, 30.0, 63.0],
        })

    def test_one_file_is_scanned_for_its_questions(self):
        root = self.folder({"emissions.csv": self.emissions()})
        found = ada.scan(root / "emissions.csv")
        self.assertEqual(len(found.files), 1)
        self.assertEqual(found.questions[0].key, "intent")
        asked = [question for question in found.questions if question.topic == "decisions"]
        keys = {question.key: question.proposed for question in asked}
        self.assertEqual(keys, {"decide:emissions.csv:aggregate_rows": "leave_out",
                                "decide:emissions.csv:total_from": "total_row"})
        self.assertIn('"decide:emissions.csv:aggregate_rows": "leave_out"', found.answers_template())

    def test_the_answers_file_carries_decisions_to_ask_and_claim(self):
        root = self.folder({"emissions.csv": self.emissions()})
        answers = root / "answers.yml"
        answers.write_text('"intent": "compare countries"\n"decide:emissions.csv:aggregate_rows": "keep"\n')
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["ask", str(root / "emissions.csv"), "average co2", "--answers", str(answers),
                         "--json"])
        self.assertEqual(code, 0)
        record = json.loads(out.getvalue())
        self.assertAlmostEqual(record["result"]["value"], 30.6)
        self.assertEqual(record["decisions"][0]["decided_by"], "caller")
        check = ada.check_claim(root / "emissions.csv", "The average co2 was 30.6.", answers=answers)
        self.assertTrue(check.verified)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(main(["ask", str(root / "emissions.csv"), "average co2", "--answers",
                                   str(root / "missing.yml")]), 2)
        self.assertIn("does not exist", err.getvalue())
        # --decide wins over the answers file.
        chosen = ada.check_claim(root / "emissions.csv", "The average co2 was 20.", answers=answers,
                                 decisions={"aggregate_rows": "leave_out"})
        self.assertTrue(chosen.verified)

    def test_answers_for_every_file_and_for_one(self):
        given = {"decide:join_links": "none", "decide:a.csv:aggregate_rows": "keep",
                 "decide:b.csv:aggregate_rows": "leave_out", "intent": "anything",
                 "rows:a.csv": "one country"}
        self.assertEqual(folder_choices(given, "a.csv", {}), {"join_links": "none", "aggregate_rows": "keep"})
        self.assertEqual(folder_choices(given, "c.csv", {"join_links": "all"}), {"join_links": "all"})
        with self.assertRaises(UnknownDecision):
            folder_choices({"decide:aggregate_rows": "drop"}, None, {})

    def test_a_file_that_is_not_a_table_is_refused(self):
        root = self.folder({"emissions.csv": self.emissions()})
        (root / "notes.json").write_text("{}")
        with self.assertRaises(NotADirectoryError):
            ada.scan(root / "notes.json")


if __name__ == "__main__":
    unittest.main()
