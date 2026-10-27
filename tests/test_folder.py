import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import ada
from ada.folder import singular


def write(folder: Path, files: dict[str, pd.DataFrame]) -> Path:
    for name, frame in files.items():
        frame.to_csv(folder / name, index=False)
    return folder


def shop() -> dict[str, pd.DataFrame]:
    """Customers, their orders and order lines, with two order lines naming an order that is not listed."""
    rng = np.random.default_rng(60)
    customers = pd.DataFrame({
        "customer_id": [f"c{n:03d}" for n in range(40)],
        "customer_zip_prefix": rng.integers(1000, 1060, 40),
        "segment": rng.choice(["retail", "trade"], 40),
    })
    orders = pd.DataFrame({
        "order_id": [f"o{n:03d}" for n in range(120)],
        "customer_id": rng.choice(customers["customer_id"], 120),
        "order_total": rng.gamma(2, 40, 120).round(2),
    })
    lines = pd.DataFrame({
        "order_id": np.r_[rng.choice(orders["order_id"], 298), ["o999", "o998"]],
        "quantity": rng.integers(1, 5, 300),
        "unit_price": rng.gamma(2, 10, 300).round(2),
    })
    places = pd.DataFrame({
        "geo_zip_prefix": np.repeat(np.arange(1000, 1060), 4),
        "lat": rng.normal(-23, 1, 240).round(4),
    })
    return {"customers.csv": customers, "orders.csv": orders, "order_lines.csv": lines, "places.csv": places}


class FolderTestCase(unittest.TestCase):
    def folder(self, files: dict[str, pd.DataFrame]) -> Path:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        return write(Path(holder.name), files)


class LinkTests(FolderTestCase):
    def test_keys_are_found_from_the_values_with_their_coverage(self):
        found = ada.scan(self.folder(shop()))
        links = {(link["from"], link["to"]): link for link in found.to_dict()["links"]}
        orders = links[("orders.customer_id", "customers.customer_id")]
        self.assertEqual((orders["kind"], orders["coverage"]), ("many-to-one", 1.0))
        self.assertEqual(orders["unmatched_rows"], 0)
        self.assertFalse(orders["needs_confirmation"])
        lines = links[("order_lines.order_id", "orders.order_id")]
        self.assertEqual((lines["unmatched_rows"], lines["unmatched_values"]), (2, 2))
        self.assertTrue(lines["needs_confirmation"])
        self.assertIn("have no match in orders.csv", lines["reasons"][0])

    def test_a_key_that_repeats_on_the_other_side_warns_of_repeated_rows(self):
        found = ada.scan(self.folder(shop()))
        link = next(link for link in found.links if link.to_columns == ("geo_zip_prefix",))
        self.assertEqual((link.kind, link.fan_out), ("many-to-many", 4))
        self.assertTrue(any("joining on it repeats rows" in reason for reason in link.reasons))

    def test_a_composite_key_and_no_link_on_one_of_its_parts(self):
        rng = np.random.default_rng(61)
        teams = pd.DataFrame({"yearID": np.repeat(np.arange(2001, 2011), 6),
                              "teamID": np.tile(["BOS", "NYA", "CHN", "SFN", "LAN", "SEA"], 10),
                              "wins": rng.integers(60, 100, 60)})
        batting = teams.sample(400, replace=True, random_state=1)[["yearID", "teamID"]].assign(
            playerID=[f"p{n}" for n in rng.integers(0, 150, 400)], HR=rng.integers(0, 40, 400))
        found = ada.scan(self.folder({"Teams.csv": teams, "Batting.csv": batting}))
        labels = {(link.label("from"), link.label("to"), link.kind) for link in found.links}
        self.assertIn(("Batting.(yearID, teamID)", "Teams.(yearID, teamID)", "many-to-one"), labels)
        self.assertFalse(any(link.from_columns == ("teamID",) for link in found.links))

    def test_names_that_differ_need_confirming(self):
        flights = pd.DataFrame({"dest": np.tile([f"A{n:02d}" for n in range(30)], 3),
                                "delay": np.arange(90.0)})
        airports = pd.DataFrame({"faa": [f"A{n:02d}" for n in range(28)] + ["ZZZ"], "alt": np.arange(29)})
        found = ada.scan(self.folder({"flights.csv": flights, "airports.csv": airports}))
        link = found.links[0]
        self.assertEqual((link.label("from"), link.label("to")), ("flights.dest", "airports.faa"))
        self.assertFalse(link.named)
        self.assertEqual(link.unmatched_values, 2)
        self.assertIn("the names differ", link.reasons[0])

    def test_whole_numbers_link_only_under_matching_names(self):
        # Grid positions 1 to 30 sit inside the driver ids, and "grid" ends in "id".
        rng = np.random.default_rng(62)
        drivers = pd.DataFrame({"driverId": np.arange(1, 61), "born": rng.integers(1950, 2000, 60)})
        results = pd.DataFrame({"driverId": rng.integers(1, 61, 400), "grid": rng.integers(1, 31, 400),
                                "points": rng.integers(0, 26, 400)})
        found = ada.scan(self.folder({"drivers.csv": drivers, "results.csv": results}))
        self.assertEqual([(link.label("from"), link.label("to")) for link in found.links],
                         [("results.driverId", "drivers.driverId")])

    def test_an_id_column_links_to_the_bare_id_of_the_file_it_names(self):
        rng = np.random.default_rng(64)
        customers = pd.DataFrame({"id": np.arange(1, 61), "name": [f"n{n}" for n in range(60)]})
        orders = pd.DataFrame({"order_id": np.arange(500), "customer_id": rng.integers(1, 61, 500),
                               "total": rng.gamma(2, 40, 500).round(2)})
        found = ada.scan(self.folder({"customers.csv": customers, "orders.csv": orders}))
        link = found.links[0]
        self.assertEqual((link.label("from"), link.label("to"), link.named),
                         ("orders.customer_id", "customers.id", True))

    def test_two_bare_ids_link_only_when_they_number_the_same_things(self):
        colors = pd.DataFrame({"id": np.arange(1, 61), "name": [f"colour {n}" for n in range(60)]})
        sets = pd.DataFrame({"id": np.arange(1, 2001), "pieces": np.arange(2000) % 300})
        found = ada.scan(self.folder({"colors.csv": colors, "sets.csv": sets}))
        self.assertEqual(found.links, ())
        # Two halves of one table of board games, each listing nearly every game.
        ratings = pd.DataFrame({"id": np.arange(1, 501), "average": np.linspace(5, 9, 500)})
        details = pd.DataFrame({"id": np.arange(3, 501), "players": np.arange(498) % 6 + 1})
        found = ada.scan(self.folder({"ratings.csv": ratings, "details.csv": details}))
        self.assertEqual([(link.label("from"), link.label("to")) for link in found.links],
                         [("details.id", "ratings.id")])

    def test_a_shared_label_is_not_a_key(self):
        nations = [f"Nation {n}" for n in range(25)]
        teams = pd.DataFrame({"team_id": np.arange(75), "nationality": np.tile(nations, 3)})
        drivers = pd.DataFrame({"driver_id": np.arange(200), "nationality": np.resize(nations, 200)})
        found = ada.scan(self.folder({"teams.csv": teams, "drivers.csv": drivers}))
        self.assertEqual(found.links, ())

    def test_a_missing_value_written_as_backslash_n_is_not_a_key_value(self):
        # MySQL exports write a missing value as \N. Read as text in a column that is mostly
        # missing, it hid that the rest are numbers, which then linked to any id column.
        rng = np.random.default_rng(63)
        races = pd.DataFrame({"raceId": np.arange(1, 41), "round": np.tile(np.arange(1, 21), 2)})
        laps = rng.integers(1, 41, 300).astype(object)
        laps[np.arange(300) % 4 != 0] = "\\N"
        results = pd.DataFrame({"raceId": rng.integers(1, 41, 300), "fastestLap": laps})
        found = ada.scan(self.folder({"races.csv": races, "results.csv": results}))
        self.assertEqual([(link.label("from"), link.label("to")) for link in found.links],
                         [("results.raceId", "races.raceId")])

    def test_tables_of_the_same_length_with_no_key_get_a_warning(self):
        rng = np.random.default_rng(62)
        found = ada.scan(self.folder({
            "a.csv": pd.DataFrame({"mass": rng.normal(4000, 500, 50)}),
            "b.csv": pd.DataFrame({"depth": rng.normal(17, 2, 50)}),
        }))
        self.assertEqual(found.links, ())
        self.assertIn("joining them by row order is only safe", found.warnings[0])


class ProfileTests(FolderTestCase):
    def test_a_codebook_gives_units_and_is_not_joined(self):
        rng = np.random.default_rng(63)
        data = pd.DataFrame({
            "country": np.repeat(["A", "B", "C"], 30), "year": np.tile(np.arange(1990, 2020), 3),
            "co2": rng.gamma(2, 10, 90).round(3),
        })
        book = pd.DataFrame({
            "column": ["country", "year", "co2"], "description": ["Place", "Year", "Emissions"],
            "unit": ["", "", "million tonnes"],
        })
        found = ada.scan(self.folder({"data.csv": data, "codebook.csv": book}))
        self.assertEqual(found.file("codebook.csv").describes, "data.csv")
        self.assertEqual(dict(found.file("data.csv").units)["co2"], "million tonnes")
        self.assertFalse(any(question.topic == "units" for question in found.questions))
        self.assertFalse(any("codebook" in f"{link.from_file} {link.to_file}" for link in found.links))

    def test_rows_are_named_by_the_file_or_by_a_row_key(self):
        found = ada.scan(self.folder(shop()))
        rows = {question.key: question.proposed for question in found.questions if question.topic == "rows"}
        self.assertEqual(rows["rows:customers.csv"], "one customer (from the file's name)")
        self.assertEqual(found.file("orders.csv").row_key, ("order_id",))

    def test_singulars(self):
        for plural, one in (("people", "person"), ("salaries", "salary"), ("statuses", "status"),
                            ("flights", "flight"), ("boxes", "box"), ("glass", "glass")):
            with self.subTest(plural=plural):
                self.assertEqual(singular(plural), one)


class AnswerTests(FolderTestCase):
    def test_answers_are_kept_and_read_back(self):
        folder = self.folder(shop())
        found = ada.scan(folder)
        template = folder / "answers.yml"
        template.write_text(found.answers_template())
        key = next(question.key for question in found.questions if question.topic == "links")
        given = re.sub(rf'"{re.escape(key)}": ".*"', f'"{key}": "no"', template.read_text())
        given = given.replace('"intent": ""', '"intent": "which customers order most"')
        template.write_text(given)
        again = ada.scan(folder, answers=template)
        self.assertNotIn(key, [link.key for link in again.links])
        intent = next(question for question in again.questions if question.key == "intent")
        self.assertEqual(intent.answer, "which customers order most")
        self.assertLess(len(again.open_questions()), len(found.open_questions()))

    def test_a_confirmed_link_is_no_longer_asked(self):
        folder = self.folder(shop())
        key = "link:order_lines.order_id->orders.order_id"
        found = ada.scan(folder, answers={key: "yes"})
        link = next(link for link in found.links if link.key == key)
        self.assertTrue(link.confirmed)
        self.assertNotIn(key, [question.key for question in found.questions])


class ThinFolderTests(FolderTestCase):
    def test_one_file_an_unreadable_file_and_an_empty_folder(self):
        folder = self.folder({"only.csv": pd.DataFrame({"x": [1.5, 2.5]})})
        (folder / "broken.csv").write_text("")
        found = ada.scan(folder)
        self.assertEqual(found.links, ())
        self.assertIsNotNone(found.file("broken.csv").problem)
        self.assertIn("only.csv: 2 rows x 1 column,", str(found))
        json.dumps(found.to_dict(), allow_nan=False)
        empty = tempfile.TemporaryDirectory()
        self.addCleanup(empty.cleanup)
        self.assertEqual(ada.scan(empty.name).files, ())
        self.assertEqual([profile.name for profile in ada.scan(folder / "only.csv").files], ["only.csv"])
        with self.assertRaises(NotADirectoryError):
            ada.scan(folder / "missing.csv")


class FolderClaimTests(FolderTestCase):
    def test_a_claim_about_a_folder_is_checked_against_the_file_it_fits(self):
        files = shop()
        folder = self.folder(files)
        total = files["orders.csv"]["order_total"].mean()
        check = ada.check_claim(folder, f"The average order_total is {total:.2f}.")
        self.assertTrue(check.verified)
        self.assertEqual(check.source.name, "orders.csv")
        self.assertIn("the only file", check.assumptions[0])

    def test_the_files_row_noun_is_not_part_of_a_value(self):
        # "Expeditions" names the rows of expeditions.csv, not the agencies called "... Expeditions".
        rng = np.random.default_rng(65)
        expeditions = pd.DataFrame({
            "expedition_id": [f"E{n:03d}" for n in range(40)],
            "trekking_agency": rng.choice(["Himalaya Expeditions", "Everest Expeditions", "Summit Club"], 40),
            "member_deaths": rng.integers(0, 3, 40),
        })
        folder = self.folder({"expeditions.csv": expeditions})
        deaths = int(expeditions["member_deaths"].sum())
        check = ada.check_claim(folder / "expeditions.csv", f"Expeditions recorded {deaths} member deaths.")
        self.assertTrue(check.verified)
        self.assertEqual(check.answer.calculation, "sum(member_deaths)")

    def test_the_file_whose_column_the_claim_names_wins_over_a_lesser_word(self):
        # "deaths" is the last word of member_deaths and only the first of death_height_metres.
        rng = np.random.default_rng(66)
        events = pd.DataFrame({"event": [f"e{n}" for n in range(30)],
                               "member_deaths": rng.integers(0, 3, 30)})
        climbers = pd.DataFrame({
            "climber": [f"c{n}" for n in range(90)],
            "death_height_metres": np.where(np.arange(90) % 15 == 0, 7000.0, np.nan),
        })
        folder = self.folder({"events.csv": events, "climbers.csv": climbers})
        deaths = int(events["member_deaths"].sum())
        check = ada.check_claim(folder, f"Total deaths were {deaths}.")
        self.assertTrue(check.verified)
        self.assertEqual(check.answer.plan.measure, "member_deaths")
        self.assertIn("climbers.csv fit only by a lesser word", check.assumptions[0])

    def test_a_claim_no_single_file_holds_names_the_joins(self):
        folder = self.folder(shop())
        check = ada.check_claim(folder, "Retail customers ordered 4,000 in total.")
        self.assertEqual(check.status, "unverified")
        self.assertIn(check.reason, ("missing_context", "unsupported_analysis", "ambiguous_intent"))


class ScanCommandTests(FolderTestCase):
    def test_json_and_template(self):
        folder = self.folder(shop())
        template = folder / "answers.yml"
        result = subprocess.run(
            [sys.executable, "-m", "ada.cli", "scan", str(folder), "--json", "--template", str(template)],
            capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["schema_version"], 1)
        topics = ["intent", "rows", "units", "repeated", "links", "decisions"]
        self.assertEqual(list(data["questions"]), topics)
        self.assertIn('"intent":', template.read_text())


if __name__ == "__main__":
    unittest.main()
