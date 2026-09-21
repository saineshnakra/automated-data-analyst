"""The backtest scores the holdout it says it does, and the note says what it measured.

Three defects from an independent review of the forecast contract, one test
each plus the corners they open up:

* a held-out period was scored at the next consecutive slot after training
  rather than at its own calendar position, so a gap inside the holdout
  shifted every later prediction one period early;
* the note claimed the model was "better than assuming no change" from
  MASE alone, whose denominator is the training history, not a no-change
  forecast on the holdout;
* the percentage error was written with a plus-minus sign, which reads as
  an uncertainty interval.
"""

import unittest

import numpy as np
import pandas as pd

from forecasting import Backtest, build_forecast, describe_backtest


def make_trend(values, freq="MS", start="2024-01-01"):
    periods = pd.date_range(start, periods=len(values), freq=freq)
    return pd.DataFrame({"Period": periods, "Value": [float(value) for value in values]})


class GappedHoldoutTests(unittest.TestCase):
    def test_a_missing_month_inside_the_holdout_is_scored_at_its_own_position(self):
        """A perfectly linear series has zero error whatever months are missing."""
        trend = make_trend([100 + 10 * index for index in range(16)])
        trend = trend.drop(index=14).reset_index(drop=True)

        forecast = build_forecast(trend)

        assert forecast is not None
        self.assertEqual(forecast.backtest.holdout_periods, 3)
        self.assertEqual(forecast.backtest.mape, 0.0)
        self.assertEqual(forecast.backtest.mase, 0.0)

    def test_a_gapless_holdout_scores_exactly_as_before(self):
        forecast = build_forecast(make_trend([100 + 10 * index for index in range(16)]))

        assert forecast is not None
        self.assertEqual(forecast.backtest.mape, 0.0)
        self.assertEqual(forecast.backtest.mase, 0.0)


class HoldoutContestTests(unittest.TestCase):
    def test_the_no_change_forecast_is_scored_on_the_same_holdout(self):
        """Training ends at 210; the holdout is 220, 230, 250, so carrying 210 errs 9.7%."""
        trend = make_trend([100 + 10 * index for index in range(16)])
        trend = trend.drop(index=14).reset_index(drop=True)

        forecast = build_forecast(trend)

        assert forecast is not None
        self.assertEqual(forecast.backtest.holdout_naive_mape, 9.7)
        self.assertIs(forecast.backtest.beats_naive_on_holdout, True)

    def test_mase_and_the_holdout_contest_answer_different_questions(self):
        """A zig-zag history moves 40 a step, so a 20-point miss is a MASE of 0.5.

        The holdout sits exactly at the last training value, so a no-change
        forecast is perfect there and the model loses that contest. The old
        note would have called this "better than assuming no change".
        """
        forecast = build_forecast(make_trend([100, 140, 100, 140, 100, 140, 100, 140, 140, 140, 140]))

        assert forecast is not None
        self.assertEqual(forecast.backtest.mase, 0.5)
        self.assertEqual(forecast.backtest.mape, 14.3)
        self.assertEqual(forecast.backtest.holdout_naive_mape, 0.0)
        self.assertIs(forecast.backtest.beats_naive_on_holdout, False)
        self.assertIn("so the model did not beat it", describe_backtest(forecast.backtest))

    def test_a_random_walk_loses_the_holdout_contest(self):
        rng = np.random.default_rng(6)
        forecast = build_forecast(make_trend(500 + np.cumsum(rng.normal(0, 60, 26))))

        assert forecast is not None
        self.assertEqual(forecast.backtest.mape, 17.1)
        self.assertEqual(forecast.backtest.holdout_naive_mape, 11.2)
        self.assertIs(forecast.backtest.beats_naive_on_holdout, False)

    def test_a_zero_holdout_has_no_contest_to_report(self):
        forecast = build_forecast(make_trend([40, 44, 39, 41, 45, 38, 42, 40, 43, 0, 0, 0]))

        assert forecast is not None
        self.assertIsNone(forecast.backtest.holdout_naive_mape)
        self.assertIsNone(forecast.backtest.beats_naive_on_holdout)
        self.assertEqual(forecast.backtest.mase, 11.06)

    def test_the_backtest_no_longer_claims_a_contest_from_mase_alone(self):
        self.assertFalse(hasattr(Backtest, "beats_no_change"))


class ErrorNoteWordingTests(unittest.TestCase):
    def test_the_note_reports_an_average_error_not_an_interval(self):
        note = describe_backtest(
            Backtest(mape=1.7, mase=0.72, holdout_periods=4, periods_without_mape=0, holdout_naive_mape=4.9)
        )

        self.assertEqual(
            note,
            "average error of 1.7% on the last 4 held-out periods, "
            "assuming no change would have erred 4.9% on the same periods, so the model beat it, "
            "MASE 0.72 (error relative to a one-step no-change baseline on the training history)",
        )
        self.assertNotIn("±", note)
        self.assertNotIn("than assuming no change", note)

    def test_the_note_counts_the_periods_a_percentage_could_be_taken_on(self):
        note = describe_backtest(
            Backtest(mape=3.3, mase=0.67, holdout_periods=5, periods_without_mape=2, holdout_naive_mape=10.0)
        )

        self.assertTrue(
            note.startswith("average error of 3.3% on the last 5 held-out periods (the 3 of them above zero)")
        )

    def test_the_note_without_a_percentage_still_names_the_holdout(self):
        backtest = Backtest(
            mape=None, mase=11.06, holdout_periods=3, periods_without_mape=3, holdout_naive_mape=None
        )

        self.assertEqual(
            describe_backtest(backtest),
            "held out the last 3 periods, every held-out period was zero, "
            "so a percentage error says nothing, "
            "MASE 11.06 (error relative to a one-step no-change baseline on the training history)",
        )

    def test_the_uncertainty_band_is_described_separately_from_the_error(self):
        rng = np.random.default_rng(8)
        forecast = build_forecast(make_trend(600 + 5 * np.arange(20) + rng.normal(0, 30, 20)))

        assert forecast is not None
        self.assertIn("band = ±", forecast.method)
        self.assertNotIn("band", describe_backtest(forecast.backtest))


if __name__ == "__main__":
    unittest.main()
