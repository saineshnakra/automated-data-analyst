"""The eight defects a review of the consistency release found.

Every one of them passed the 385 tests that shipped with the change. They are
here so the tests and the behaviour are the same size: a suite that is green
against a forecast reading "nan" is a suite measuring the wrong thing.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from aggregation import build_trend
from ai_insights import _usable_dimension
from analysis import _read_formatted_number
from anomalies import detect_anomalies
from autovis import ChartSpec
from forecasting import build_forecast
from nlq import answer_question
from pipeline import prepare_analysis
from schema import detect_roles
from ui import _explore_frame


def monthly(values, *, measure="Revenue", start="2022-01-31"):
    dates = pd.date_range(start, periods=len(values), freq="ME")
    return pd.DataFrame({"Date": dates, measure: values})


class UnobservedPeriodTests(unittest.TestCase):
    """A blank month is not a number the arithmetic can carry."""

    def test_one_blank_month_does_not_make_the_whole_forecast_nan(self):
        values = [100.0 + 5 * i for i in range(24)]
        values[7] = np.nan
        frame = monthly(values)
        series = build_trend(frame, detect_roles(frame))
        self.assertTrue(series.frame["Value"].isna().any(), "fixture must contain the gap")

        forecast = build_forecast(series.frame)
        self.assertIsNotNone(forecast)
        for name in ("values", "lower", "upper"):
            numbers = np.asarray(getattr(forecast, name), dtype=float)
            self.assertFalse(np.isnan(numbers).any(), f"{name} carries NaN")

    def test_a_spike_is_still_found_when_a_month_is_blank(self):
        values = [100.0] * 24
        values[5] = np.nan
        values[18] = 1000.0
        frame = monthly(values)
        series = build_trend(frame, detect_roles(frame))
        found = detect_anomalies(series.frame)
        self.assertTrue(found, "an obvious spike went unreported")

    def test_a_rate_with_a_missing_month_still_forecasts(self):
        # The other path into NaN: rate gaps are filled with NaN by design.
        dates = [d for d in pd.date_range("2022-01-31", periods=24, freq="ME")]
        del dates[9]
        frame = pd.DataFrame({"Date": dates, "Conversion Rate": [0.2 + 0.001 * i for i in range(23)]})
        series = build_trend(frame, detect_roles(frame))
        self.assertEqual(series.filled_periods, 1)

        forecast = build_forecast(series.frame)
        self.assertIsNotNone(forecast)
        self.assertFalse(np.isnan(np.asarray(forecast.values, dtype=float)).any())

    def test_the_caption_does_not_call_a_rate_gap_a_zero(self):
        dates = [d for d in pd.date_range("2022-01-31", periods=12, freq="ME")]
        del dates[4]
        rates = pd.DataFrame({"Date": dates, "Conversion Rate": [0.2] * 11})
        note = " ".join(build_trend(rates, detect_roles(rates)).notes)
        self.assertIn("left empty", note)
        self.assertNotIn("counted as zero", note)

        # An amount still says zero, because for an amount that is true.
        amounts = pd.DataFrame({"Date": dates, "Revenue": [100.0] * 11})
        amount_note = " ".join(build_trend(amounts, detect_roles(amounts)).notes)
        self.assertIn("counted as zero", amount_note)


class TrimTests(unittest.TestCase):
    def test_a_row_cap_never_empties_a_single_period_file(self):
        # A quarter of a million rows from one busy week: every kept row sits
        # in the one truncated period, and dropping it left nothing at all.
        dates = pd.to_datetime(["2025-03-03"] * 400)
        frame = pd.DataFrame({"Date": dates, "Revenue": np.arange(400.0)})
        prepared = prepare_analysis(frame, row_limit=100)
        self.assertEqual(len(prepared.dataframe), 100)
        self.assertGreater(prepared.dataframe["Revenue"].sum(), 0)


class FilterTests(unittest.TestCase):
    def test_a_filter_matches_every_spelling_of_the_value(self):
        # "West" and "west" are one word to the parser and two rows to pandas.
        frame = pd.DataFrame(
            {
                "Region": ["West"] * 15 + ["west"] * 15 + ["East"] * 10,
                "Revenue": [10.0] * 40,
            }
        )
        answer = answer_question("total revenue for west", frame, detect_roles(frame))
        self.assertIsNotNone(answer)
        self.assertIn("300", answer.answer.replace(",", ""))


class NumberTests(unittest.TestCase):
    def test_a_parenthesised_number_stays_negative_whatever_leads_it(self):
        self.assertEqual(_read_formatted_number("(+100)"), -100.0)
        self.assertEqual(_read_formatted_number("(100)"), -100.0)
        self.assertEqual(_read_formatted_number("(-100)"), -100.0)
        self.assertEqual(_read_formatted_number("-100"), -100.0)
        self.assertEqual(_read_formatted_number("100"), 100.0)


class DimensionGuardTests(unittest.TestCase):
    def test_a_unique_text_column_is_not_a_segment_even_without_code_shapes(self):
        # Sixty distinct emails group into sixty rows of one - the table
        # again, with a chart drawn on it.
        emails = pd.DataFrame({"Customer Email": [f"person{i}@example.com" for i in range(60)]})
        self.assertFalse(_usable_dimension(emails, "Customer Email"))
        # A real segment still passes.
        regions = pd.DataFrame({"Region": ["North", "South"] * 30})
        self.assertTrue(_usable_dimension(regions, "Region"))


class ChartNameTests(unittest.TestCase):
    def test_a_measure_called_records_survives_a_histogram(self):
        frame = pd.DataFrame({"Records": np.random.default_rng(3).normal(100, 10, 500)})
        binned = _explore_frame(frame, ChartSpec(form="histogram", rationale="", x="Records"))
        counts = binned.attrs["count_column"]
        self.assertNotEqual(counts, "Records")
        self.assertEqual(int(binned[counts].sum()), 500)
        # The bin edges are the measure, not the counts.
        self.assertLess(binned["Records"].max(), 200)


if __name__ == "__main__":
    unittest.main()
