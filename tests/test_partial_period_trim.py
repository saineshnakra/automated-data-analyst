"""The row cap must not invent a collapse, or an empty file.

A large upload is trimmed to its most recent rows. When that cut lands inside
a period, the oldest kept period is half-present - a February holding one of
its two records reads as 100 against March's 200, and the brief reports a
doubling that never happened.

The second half is the guard on the first. Dropping that period is right when
there is something behind it and catastrophic when there is not: a quarter of a
million rows all stamped within one busy week are ALL in the truncated period,
and removing it leaves nothing to analyze at all.
"""

from __future__ import annotations

import unittest

import pandas as pd

from pipeline import prepare_analysis


def dated(pairs):
    frame = pd.DataFrame(pairs, columns=["Date", "Revenue"])
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


class PartialPeriodTests(unittest.TestCase):
    def test_a_half_present_oldest_period_is_dropped(self):
        # Two records a month for eight months. Keeping fifteen of sixteen
        # cuts one February record, so February is partial.
        rows = [(f"2025-{month:02d}-{day:02d}", 100.0) for month in range(2, 10) for day in (10, 20)]
        prepared = prepare_analysis(dated(rows), row_limit=15)
        self.assertEqual(prepared.analyzed_from, pd.Timestamp("2025-03-10"))
        self.assertNotIn(2, prepared.dataframe["Date"].dt.month.unique())

    def test_a_period_that_survives_the_cut_whole_is_kept(self):
        # Sixteen of sixteen: nothing was cut, so nothing is dropped.
        rows = [(f"2025-{month:02d}-{day:02d}", 100.0) for month in range(2, 10) for day in (10, 20)]
        prepared = prepare_analysis(dated(rows), row_limit=16)
        self.assertEqual(prepared.analyzed_from, pd.Timestamp("2025-02-10"))

    def test_a_file_that_is_all_one_period_is_not_emptied(self):
        # The guard. Every kept row is in the truncated period; dropping it
        # would report an empty dataset and a $0.00 headline for 400 rows.
        rows = [("2025-03-03", float(n)) for n in range(400)]
        prepared = prepare_analysis(dated(rows), row_limit=100)
        self.assertEqual(len(prepared.dataframe), 100)
        self.assertGreater(prepared.dataframe["Revenue"].sum(), 0)

    def test_a_file_under_the_cap_is_untouched(self):
        rows = [(f"2025-0{month}-10", 100.0) for month in range(1, 6)]
        prepared = prepare_analysis(dated(rows), row_limit=1000)
        self.assertEqual(len(prepared.dataframe), 5)
        self.assertEqual(prepared.analyzed_from, pd.Timestamp("2025-01-10"))

    def test_a_file_with_no_dates_still_takes_its_tail(self):
        frame = pd.DataFrame({"Segment": list("abcdefghij"), "Revenue": [float(n) for n in range(10)]})
        prepared = prepare_analysis(frame, row_limit=4)
        self.assertEqual(len(prepared.dataframe), 4)
        self.assertEqual(prepared.dataframe["Revenue"].tolist(), [6.0, 7.0, 8.0, 9.0])


if __name__ == "__main__":
    unittest.main()
