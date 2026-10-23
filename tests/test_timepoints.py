import unittest

import numpy as np
import pandas as pd

import ada
from ada.business_insights import analyze_business
from ada.nlq import answer_question
from ada.schema import ColumnRoles, detect_roles
from ada.timepoints import (
    is_timepoint_name,
    series_columns,
    summarize_series,
    time_profile,
    timepoint_column,
)


def infection(seed: int = 0) -> pd.DataFrame:
    """Mice measured daily: the Dead group stops at day 10, the Live group recovers to day 15."""
    rng = np.random.default_rng(seed)
    rows = []
    for mouse in range(12):
        group = "Dead" if mouse < 6 else "Live"
        last = 10 if group == "Dead" else 15
        for day in range(last + 1):
            level = 5.0 if group == "Dead" else 3.0 + (4.0 * (day - 10) if day > 10 else 0.0)
            rows.append((f"m{mouse}", group, day, level + rng.normal(0, 0.5)))
    return pd.DataFrame(rows, columns=["indiv", "group", "timept", "x"])


def roles_of(frame: pd.DataFrame) -> ColumnRoles:
    return ColumnRoles(
        date=None, measure="x", dimension="group", identifier=None,
        numeric=("timept", "x"), dimensions=("group",),
    )


class TimepointColumnTests(unittest.TestCase):
    def test_days_since_infection_are_a_time_axis(self):
        frame = infection()

        self.assertEqual(timepoint_column(frame, roles_of(frame)), "timept")

    def test_a_day_beside_a_date_is_a_part_of_it(self):
        frame = infection().assign(Date=pd.Timestamp("2024-01-01"))
        roles = ColumnRoles(date="Date", measure="x", dimension="group", identifier=None,
                            numeric=("timept", "x"), dimensions=("group",))

        self.assertIsNone(timepoint_column(frame, roles))

    def test_rates_minimums_and_totals_are_not_named_like_time(self):
        for name in ("timept", "Day", "hours_post_dose", "days_since_infection", "TimePoint"):
            self.assertTrue(is_timepoint_name(name), name)
        for name in ("drinks_day", "welding speed (mm/min)", "pollutant_min", "T_Ca", "reaction_time"):
            self.assertFalse(is_timepoint_name(name), name)

    def test_a_reading_named_like_time_with_unique_values_is_not_an_axis(self):
        frame = pd.DataFrame({"reaction_time": np.linspace(0.2, 0.9, 50), "score": np.arange(50.0)})

        self.assertIsNone(timepoint_column(frame, detect_roles(frame)))


class ProfileTests(unittest.TestCase):
    def test_groups_followed_for_different_lengths_are_compared_where_they_overlap(self):
        frame = infection()

        profile = time_profile(frame, roles_of(frame), "timept")

        assert profile is not None and profile.reach is not None
        self.assertEqual(profile.reach.shared, (0.0, 10.0))
        shared = dict(profile.reach.shared_means)
        pooled = dict(profile.reach.pooled_means)
        self.assertGreater(shared["Dead"], shared["Live"])
        self.assertGreater(pooled["Live"], shared["Live"])

    def test_the_brief_compares_the_shared_stretch_and_not_the_axis(self):
        frame = infection()

        brief = analyze_business(frame, roles_of(frame))

        profile = next(item for item in brief.evidence if item.kind == "profile")
        self.assertTrue(brief.headline.startswith("The groups were followed for different lengths"))
        self.assertIn("in order across its 16 time points", profile.statement)
        # Each mouse's clock is known, so the group card compares the shared stretch too.
        leader = next(item for item in brief.evidence if item.kind == "leader")
        self.assertIn("followed for different lengths", leader.statement)

    def test_pooled_means_of_unequal_follow_up_are_not_a_group_finding(self):
        frame = infection().drop(columns="indiv")

        brief = analyze_business(frame, roles_of(frame))

        self.assertNotIn("leader", [item.kind for item in brief.evidence])
        differences = next((item for item in brief.evidence if item.kind == "differences"), None)
        if differences is not None:
            self.assertNotIn("timept by group", differences.statement)

    def test_a_cycle_is_not_called_a_rise(self):
        phase = np.tile(np.arange(1, 100) / 100, 20)
        frame = pd.DataFrame({"time": phase, "angle": np.sin(2 * np.pi * phase) + 0.01 * phase})
        roles = ColumnRoles(date=None, measure="angle", dimension=None, identifier=None,
                            numeric=("time", "angle"), dimensions=())

        profile = next(item for item in analyze_business(frame, roles).evidence if item.kind == "profile")

        self.assertIn("ends about where it started", profile.statement)
        self.assertNotIn("rises", profile.statement)


class SeriesColumnTests(unittest.TestCase):
    def time_course(self, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        minutes = np.arange(40, 270, 10)
        rows = rng.normal(0, 0.3, (300, len(minutes)))
        rows[:60] += 1.5 * np.sin(2 * np.pi * minutes / 110)
        frame = pd.DataFrame(rows, columns=[str(minute) for minute in minutes])
        frame.insert(0, "gene", [f"G{index:03d}" for index in range(300)])
        return frame

    def test_evenly_numbered_columns_are_a_series(self):
        columns, positions = series_columns(self.time_course())

        self.assertEqual(columns[0], "40")
        self.assertEqual(len(columns), 23)
        self.assertEqual(positions[1] - positions[0], 10)
        self.assertIsNone(series_columns(self.time_course().drop(columns=["50", "60"])))

    def test_a_repeating_pattern_is_found_with_its_period(self):
        frame = self.time_course()
        roles = ColumnRoles(date=None, measure=None, dimension=None, identifier="gene",
                            numeric=tuple(frame.columns[1:]), dimensions=())
        columns, positions = series_columns(frame)

        summary = summarize_series(frame, roles, columns, positions)

        self.assertAlmostEqual(summary.period, 110, delta=15)
        self.assertGreaterEqual(summary.clear_rows, 40)
        self.assertLessEqual(summary.clear_rows, 80)
        self.assertTrue(summary.varied[0][0].startswith("G0"))


class AskProfileTests(unittest.TestCase):
    def test_a_breakdown_by_a_long_time_axis_is_shown_in_order(self):
        phase = np.tile(np.arange(1, 100) / 100, 5)
        frame = pd.DataFrame({"time": phase, "angle": np.sin(2 * np.pi * phase)})
        roles = ColumnRoles(date=None, measure="angle", dimension="time", identifier=None,
                            numeric=("time", "angle"), dimensions=("time",))

        answer = answer_question("average angle by time", frame, roles)

        self.assertEqual(len(answer.table), 99)
        self.assertTrue(answer.table.iloc[:, 0].is_monotonic_increasing)
        self.assertIn("It is lowest at time 0.75", answer.answer)


class AxisOverrideTests(unittest.TestCase):
    def test_a_numeric_date_named_like_time_is_the_axis(self):
        report = ada.analyze(infection(), date="timept")

        self.assertEqual(report.axis, "timept")
        self.assertIsNone(report.roles.date)
        self.assertIn("elapsed time", report.forecast().message)
        self.assertIn("profile", [item.kind for item in report.evidence])


if __name__ == "__main__":
    unittest.main()
