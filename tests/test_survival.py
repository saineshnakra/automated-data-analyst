import unittest

import numpy as np
import pandas as pd

from ada.business_insights import analyze_business
from ada.schema import ColumnRoles
from ada.survival import describe_survival, landmark, summarize_survival, survival_pair


def cohort(size: int = 300, seed: int = 0) -> pd.DataFrame:
    """Patients by grade: grade 3 has five times grade 1's hazard; follow-up stops at 15."""
    rng = np.random.default_rng(seed)
    grade = rng.choice([1, 2, 3], size)
    event_time = rng.exponential(40.0 / np.array([1.0, 2.5, 5.0])[grade - 1])
    censor_time = rng.uniform(2, 15, size)
    return pd.DataFrame({
        "ID": np.arange(size),
        "survival": np.minimum(event_time, censor_time).round(3),
        "eventdeath": (event_time <= censor_time).astype(int),
        "grade": grade,
        "chemo": rng.integers(0, 2, size),
        "timerecurrence": (np.minimum(event_time, censor_time) * rng.uniform(0.5, 1, size)).round(3),
    })


def roles_of(frame: pd.DataFrame, measure: str = "survival", dimension: str = "eventdeath") -> ColumnRoles:
    return ColumnRoles(
        date=None, measure=measure, dimension=dimension, identifier="ID",
        numeric=("survival", "timerecurrence"), dimensions=(dimension,),
        factors=("eventdeath", "grade", "chemo"), flags=("eventdeath", "chemo"),
    )


class SurvivalPairTests(unittest.TestCase):
    def test_a_follow_up_time_pairs_with_its_death_flag(self):
        frame = cohort()

        pair = survival_pair(frame, roles_of(frame))

        assert pair is not None
        found = (pair.time, pair.event, pair.event_value, pair.censored_value)
        self.assertEqual(found, ("survival", "eventdeath", 1, 0))

    def test_names_that_share_a_stem_pair_up(self):
        frame = cohort().rename(columns={"survival": "os_months", "eventdeath": "os_status"})

        pair = survival_pair(frame, roles_of(frame, "os_months", "os_status"))

        assert pair is not None
        self.assertEqual((pair.time, pair.event), ("os_months", "os_status"))

    def test_a_censoring_flag_is_read_the_other_way(self):
        frame = cohort()
        frame["censored"] = 1 - frame.pop("eventdeath")

        pair = survival_pair(frame, roles_of(frame, dimension="censored"))

        assert pair is not None
        self.assertEqual(pair.event_value, 0)
        self.assertEqual(pair.status(frame).sum(), (frame["censored"] == 0).sum())

    def test_a_text_status_needs_a_value_that_names_the_event(self):
        frame = cohort()
        frame["vital_status"] = np.where(frame.pop("eventdeath") == 1, "Dead", "Alive")
        self.assertEqual(survival_pair(frame, roles_of(frame, dimension="vital_status")).event_value, "Dead")

        frame["vital_status"] = np.where(frame["vital_status"] == "Dead", "Closed", "Open")
        self.assertIsNone(survival_pair(frame, roles_of(frame, dimension="vital_status")))

    def test_a_delivery_time_and_its_status_are_not_survival(self):
        frame = pd.DataFrame({
            "Delivery Time": np.linspace(1, 30, 100),
            "Status": ["Delivered", "Failed"] * 50,
        })

        self.assertIsNone(survival_pair(frame, roles_of(frame, "Delivery Time", "Status")))


class SurvivalSummaryTests(unittest.TestCase):
    def test_grades_differ_by_the_log_rank_test_with_hazard_ratios(self):
        frame = cohort(seed=1)
        roles = roles_of(frame)

        summary = summarize_survival(frame, roles, survival_pair(frame, roles))

        self.assertEqual(summary.groups[0].grouping, "grade")
        self.assertLess(summary.groups[0].adjusted_p, 0.001)
        shares = [share for _, share, _ in summary.groups[0].at_landmark]
        self.assertGreater(shares[0], shares[2])
        self.assertEqual([ratio.level for ratio in summary.hazards], [2, 3])
        self.assertGreater(summary.hazards[-1].ratio, 2)

    def test_the_statement_says_the_time_is_censored(self):
        frame = cohort(seed=2)
        roles = roles_of(frame)

        statement, lead = describe_survival(summarize_survival(frame, roles, survival_pair(frame, roles)))

        self.assertIn("survival is follow-up time, censored where eventdeath = 0", statement)
        self.assertIn("log-rank", statement)
        self.assertTrue(lead.startswith("Survival differs by grade"))

    def test_a_time_to_another_event_is_not_called_survival(self):
        frame = cohort(seed=4).rename(columns={"survival": "TIME", "eventdeath": "STATUS"})
        roles = roles_of(frame, "TIME", "STATUS")

        statement, lead = describe_survival(summarize_survival(frame, roles, survival_pair(frame, roles)))

        self.assertTrue(lead.startswith("Time to STATUS 1 differs by grade"), lead)
        self.assertNotIn("survival", statement.casefold())

    def test_the_landmark_is_a_round_time_within_follow_up(self):
        self.assertEqual(landmark(np.linspace(0.5, 14, 100)), 10.0)
        self.assertEqual(landmark(np.linspace(1, 400, 100)), 250.0)


class SurvivalBriefTests(unittest.TestCase):
    def test_the_brief_never_compares_follow_up_across_its_own_event(self):
        frame = cohort(seed=3)

        brief = analyze_business(frame, roles_of(frame))

        self.assertTrue(brief.headline.startswith("Survival differs by grade"), brief.headline)
        self.assertNotIn("leader", [item.kind for item in brief.evidence])
        differences = next((item for item in brief.evidence if item.kind == "differences"), None)
        if differences is not None:
            self.assertNotIn("survival differs across eventdeath", differences.statement)
            self.assertNotIn("timerecurrence by eventdeath", differences.statement)
        self.assertNotIn("Average survival", [kpi.label for kpi in brief.kpis])

    def test_units_followed_over_several_rows_get_no_survival_card(self):
        frame = cohort(seed=5)
        frame["Subject"] = np.repeat(np.arange(30), 10)

        brief = analyze_business(frame, roles_of(frame, dimension="grade"))

        self.assertNotIn("survival", [item.kind for item in brief.evidence])


if __name__ == "__main__":
    unittest.main()
