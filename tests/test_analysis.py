import time
import unittest

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from ada.analysis import (
    _numeric_from_text,
    _read_formatted_number,
    build_markdown_report,
    clean_dataframe,
    column_profile,
    generate_insights,
    separate_stacked_logs,
    set_aside_coarser_rows,
    CleaningSuggestion,
    apply_cleaning_suggestion,
    preview_cleaning_suggestion,
    suggest_cleaning,
)
from ada.formatting import format_moment, is_undated
from ada.schema import (
    DIMENSION_KEYWORDS,
    MEASURE_KEYWORDS,
    _keyword_score,
    detect_roles,
    looks_like_identifier,
)


class CleanDataframeTests(unittest.TestCase):
    def test_cleaning_is_conservative_and_audited(self):
        raw = pd.DataFrame(
            {
                "Unnamed: 0": [0, 1, 2, 3],
                " amount ": [" 10.5 ", "20", "20", "20"],
                "Order Date": ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-02"],
                "Region": [" West ", "East", "East", "East"],
                "Empty": [np.nan, np.nan, np.nan, np.nan],
            }
        )

        cleaned, report = clean_dataframe(raw)

        self.assertNotIn("Unnamed: 0", cleaned.columns)
        self.assertNotIn("Empty", cleaned.columns)
        self.assertEqual(cleaned.columns.tolist(), ["amount", "Order Date", "Region"])
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned["amount"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned["Order Date"]))
        self.assertEqual(cleaned.loc[0, "Region"], "West")
        self.assertEqual(report.index_columns_removed, 1)
        self.assertEqual(report.empty_columns_removed, 1)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            clean_dataframe(pd.DataFrame())

    def test_a_whitespace_only_column_is_empty(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="MS")
        frame = pd.DataFrame({"Date": dates, "Revenue": range(5), "Note": ["  "] * 5})

        self.assertNotIn("Note", clean_dataframe(frame)[0].columns)

    def test_a_date_column_is_found_whatever_it_is_called(self):
        frame = pd.DataFrame(
            {"Month": [f"1949-{month:02d}" for month in range(1, 13)], "Passengers": range(12)}
        )

        cleaned, report = clean_dataframe(frame)

        self.assertTrue(is_datetime64_any_dtype(cleaned["Month"]))
        self.assertEqual(report.datetime_columns_inferred, 1)

    def test_numbers_in_the_range_of_years_stay_numbers(self):
        frame = pd.DataFrame({"Count": ["1949", "1950", "1951"], "Passengers": [1, 2, 3]})

        cleaned, _ = clean_dataframe(frame)

        self.assertFalse(is_datetime64_any_dtype(cleaned["Count"]))

    def test_text_categories_are_not_mistaken_for_dates(self):
        frame = pd.DataFrame(
            {"Container": ["Small Box", "Jumbo Drum", "Large Box"], "Sales": [1.0, 2.0, 3.0]}
        )

        cleaned, _ = clean_dataframe(frame)

        self.assertFalse(is_datetime64_any_dtype(cleaned["Container"]))


    def test_numeric_cleanup_is_suggested_without_changing_data(self):
        frame = pd.DataFrame(
            {"Amount": ["$4,906.87", "$1,200.00", "$850.50"]}
        )

        suggestions = suggest_cleaning(frame)

        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0].column, "Amount")
        self.assertEqual(suggestions[0].operation, "parse_numeric")

    def test_numeric_cleanup_preview_contains_before_and_after_values(self):
        frame = pd.DataFrame(
            {"Amount": ["$4,906.87", "$1,200.00", "$850.50"]}
        )

        suggestions = suggest_cleaning(frame)

        preview = preview_cleaning_suggestion(frame, suggestions[0])

        self.assertEqual(
            preview["before"].tolist(),
            ["$4,906.87", "$1,200.00", "$850.50"],
        )
        self.assertEqual(
            preview["after"].tolist(),
            [4906.87, 1200.00, 850.50],
        )

    def test_identifier_columns_are_not_suggested_as_numeric(self):
        frame = pd.DataFrame(
            {"Customer ID": ["00123", "00456", "00789"]}
        )

        suggestions = suggest_cleaning(frame)

        self.assertEqual(suggestions, [])

    def test_accounting_parentheses_are_suggested_as_negative_numbers(self):
        frame = pd.DataFrame(
            {"Amount": ["4,906.87", "(208.82)", "100.00"]}
        )

        suggestions = suggest_cleaning(frame)

        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0].column, "Amount")
        self.assertEqual(suggestions[0].operation, "parse_numeric")

        preview = preview_cleaning_suggestion(frame, suggestions[0])

        self.assertEqual(
            preview["after"].tolist(),
            [4906.87, -208.82, 100.00],
        )

    def test_numeric_value_with_units_is_suggested(self):
        frame = pd.DataFrame(
            {"Quantity": ["3,450 units", "1,200 units", "850 units"]}
        )

        suggestions = suggest_cleaning(frame)

        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0].column, "Quantity")
        self.assertEqual(suggestions[0].operation, "parse_numeric")

        preview = preview_cleaning_suggestion(frame, suggestions[0])

        self.assertEqual(
            preview["after"].tolist(),
            [3450, 1200, 850],
        )

    def test_consistent_delimiter_is_suggested_for_split(self):
        frame = pd.DataFrame(
            {
                "Segment": [
                    "West / Enterprise",
                    "East / Consumer",
                    "South / Enterprise",
                ]
            }
        )

        suggestions = suggest_cleaning(frame)

        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0].column, "Segment")
        self.assertEqual(suggestions[0].operation, "split")
    def test_split_preview_shows_before_and_after_values(self):
        frame = pd.DataFrame(
            {
                "Segment": [
                    "West / Enterprise",
                    "East / Consumer",
                    "South / Enterprise",
                ]
            }
        )

        suggestions = suggest_cleaning(frame)

        preview = preview_cleaning_suggestion(frame, suggestions[0])

        self.assertEqual(
            preview["before"].tolist(),
            [
                "West / Enterprise",
                "East / Consumer",
                "South / Enterprise",
            ],
        )
        self.assertEqual(
            preview["after"].tolist(),
            [
                "West | Enterprise",
                "East | Consumer",
                "South | Enterprise",
            ],
        )

    def test_apply_numeric_cleaning_suggestion(self):
        frame = pd.DataFrame(
            {
                "Revenue": ["$4,906.87", "$1,200.00", "$850.50"],
            }
        )

        suggestion = suggest_cleaning(frame)[0]

        cleaned = apply_cleaning_suggestion(frame, suggestion)

        self.assertEqual(
            cleaned["Revenue"].tolist(),
            [4906.87, 1200.0, 850.5],
        )

    def test_apply_split_cleaning_suggestion(self):
        frame = pd.DataFrame(
            {
                "Segment": [
                    "West / Enterprise",
                    "East / Consumer",
                    "South / Enterprise",
                ]
            }
        )

        suggestion = suggest_cleaning(frame)[0]

        cleaned = apply_cleaning_suggestion(frame, suggestion)

        self.assertNotIn("Segment", cleaned.columns)

        self.assertEqual(
            cleaned["Segment_1"].tolist(),
            ["West", "East", "South"],
        )

        self.assertEqual(
            cleaned["Segment_2"].tolist(),
            ["Enterprise", "Consumer", "Enterprise"],
        )

    def test_apply_split_cleaning_suggestion_preserves_nulls(self):
        frame = pd.DataFrame(
            {
                "Segment": [
                    "West / Enterprise",
                    None,
                    "South / Enterprise",
                ]
            }
        )

        suggestion = suggest_cleaning(frame)[0]

        cleaned = apply_cleaning_suggestion(frame, suggestion)

        self.assertEqual(cleaned.loc[0, "Segment_1"], "West")
        self.assertTrue(pd.isna(cleaned.loc[1, "Segment_1"]))
        self.assertEqual(cleaned.loc[2, "Segment_1"], "South")

        self.assertEqual(cleaned.loc[0, "Segment_2"], "Enterprise")
        self.assertTrue(pd.isna(cleaned.loc[1, "Segment_2"]))
        self.assertEqual(cleaned.loc[2, "Segment_2"], "Enterprise")

    def test_apply_split_cleaning_suggestion_avoids_existing_column_names(self):
        frame = pd.DataFrame(
            {
                "Segment": [
                    "West / Enterprise",
                    "East / Consumer",
                ],
                "Segment_1": [
                    "existing value 1",
                    "existing value 2",
                ],
            }
        )

        suggestion = suggest_cleaning(frame)[0]

        cleaned = apply_cleaning_suggestion(frame, suggestion)

        self.assertIn("Segment_1", cleaned.columns)
        self.assertIn("Segment_1_2", cleaned.columns)
        self.assertIn("Segment_2", cleaned.columns)

        self.assertEqual(
            cleaned["Segment_1"].tolist(),
            ["existing value 1", "existing value 2"],
        )

        self.assertEqual(
            cleaned["Segment_1_2"].tolist(),
            ["West", "East"],
        )

        self.assertEqual(
            cleaned["Segment_2"].tolist(),
            ["Enterprise", "Consumer"],
        )

    def test_numeric_cleaning_suggestion_allows_consistent_currency(self):
        frame = pd.DataFrame(
            {
                "Revenue": ["$100", "$200", "$300"],
            }
        )

        suggestions = suggest_cleaning(frame)

        self.assertTrue(
            any(
                suggestion.column == "Revenue"
                and suggestion.operation == "parse_numeric"
                for suggestion in suggestions
            )
        )


    def test_numeric_cleaning_suggestion_rejects_mixed_currencies(self):
        frame = pd.DataFrame(
            {
                "Revenue": ["$100", "\u20ac200", "$300"],
            }
        )

        suggestions = suggest_cleaning(frame)

        self.assertFalse(
            any(
                suggestion.column == "Revenue"
                and suggestion.operation == "parse_numeric"
                for suggestion in suggestions
            )
        )


    def test_numeric_values_with_thousands_separator_are_not_split(self):
        frame = pd.DataFrame({"Revenue": ["1,234", "5,678", "9,012"]})

        suggestions = suggest_cleaning(frame)

        self.assertTrue(
            any(
                s.column == "Revenue" and s.operation == "parse_numeric"
                for s in suggestions
            )
        )
        self.assertFalse(
            any(
                s.column == "Revenue" and s.operation == "split"
                for s in suggestions
            )
        )

    def test_split_preview_handles_null_values(self):
        frame = pd.DataFrame(
            {"Segment": ["West / Enterprise", None, "South / Enterprise"]}
        )
        suggestion = CleaningSuggestion(
            column="Segment",
            operation="split",
            description="Split values",
        )

        preview = preview_cleaning_suggestion(frame, suggestion)

        self.assertEqual(preview.loc[0, "after"], "West | Enterprise")
        self.assertTrue(pd.isna(frame.loc[1, "Segment"]))
        self.assertEqual(preview.loc[1, "after"], " | ")
        self.assertEqual(preview.loc[2, "after"], "South | Enterprise")

    def test_paid_column_is_not_excluded_by_protected_name_filter(self):
        frame = pd.DataFrame({"Paid": ["100", "200", "300"]})

        suggestions = suggest_cleaning(frame)

        self.assertTrue(
            any(
                s.column == "Paid" and s.operation == "parse_numeric"
                for s in suggestions
            )
        )

    def test_numeric_parser_handles_negative_and_accounting_values(self):
        frame = pd.DataFrame({"Revenue": ["-1,200", "(2,500)", "300"]})

        suggestions = suggest_cleaning(frame)
        numeric_suggestion = next(
            s for s in suggestions
            if s.column == "Revenue" and s.operation == "parse_numeric"
        )

        cleaned = apply_cleaning_suggestion(frame, numeric_suggestion)

        self.assertEqual(cleaned["Revenue"].tolist(), [-1200.0, -2500.0, 300.0])



class YearColumnTests(unittest.TestCase):
    def test_a_year_column_becomes_the_first_of_each_year(self):
        frame = pd.DataFrame({"Year": ["1949", "1950", "1951"], "Passengers": [1, 2, 3]})

        cleaned, report = clean_dataframe(frame)

        self.assertTrue(is_datetime64_any_dtype(cleaned["Year"]))
        self.assertEqual(cleaned["Year"].tolist(), list(pd.to_datetime(["1949", "1950", "1951"])))
        self.assertEqual(report.datetime_columns_inferred, 1)
        self.assertTrue(any("Year holds whole years" in note for note in report.notes))

    def test_a_month_column_makes_the_dates_monthly(self):
        for months in (["January", "February"] * 2, ["Jan", "feb"] * 2, [1, 2] * 2):
            with self.subTest(months=months):
                frame = pd.DataFrame(
                    {"year": [1949, 1949, 1950, 1950], "month": months, "passengers": [1, 2, 3, 4]}
                )

                cleaned, report = clean_dataframe(frame)

                self.assertEqual(cleaned.columns.tolist(), ["year-month", "passengers"])
                self.assertEqual(cleaned["year-month"].iloc[3], pd.Timestamp("1950-02-01"))
                self.assertTrue(any("year and month were combined" in note for note in report.notes))

    def _hourly(self, **names) -> pd.DataFrame:
        """Six hourly readings across a new year, with the date in four columns named by ``names``."""
        stamps = pd.date_range("2016-12-31 21:00", periods=6, freq="h")
        parts = {"year": stamps.year, "month": stamps.month, "day": stamps.day, "hour": stamps.hour}
        frame = pd.DataFrame({names.get(part, part): values for part, values in parts.items()})
        frame["PM2.5"] = [4.0, 8.0, 7.0, 6.0, 3.0, 5.0]
        return frame

    def test_day_and_hour_columns_make_hourly_timestamps_beside_the_parts(self):
        cleaned, report = clean_dataframe(self._hourly())

        self.assertEqual(
            cleaned.columns.tolist(), ["year-month-day-hour", "year", "month", "day", "hour", "PM2.5"]
        )
        self.assertEqual(
            cleaned["year-month-day-hour"].tolist(),
            list(pd.date_range("2016-12-31 21:00", periods=6, freq="h")),
        )
        self.assertEqual(cleaned["hour"].tolist(), [21, 22, 23, 0, 1, 2])
        self.assertEqual(report.datetime_columns_inferred, 1)
        self.assertIn(
            "year, month, day and hour were combined into one timestamp column, year-month-day-hour, "
            "to the hour, with no time zone attached; the original columns are kept.",
            report.notes,
        )

    def test_a_day_column_makes_daily_dates(self):
        frame = self._hourly().drop(columns=["hour"])

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(cleaned["year-month-day"].dt.day.tolist(), [31, 31, 31, 1, 1, 1])
        self.assertIn("day", cleaned.columns)
        self.assertTrue(any("to the day" in note for note in report.notes))

    def test_an_hour_without_a_day_leaves_the_dates_monthly(self):
        frame = self._hourly().drop(columns=["day"])

        cleaned, _ = clean_dataframe(frame)

        self.assertEqual(cleaned.columns.tolist(), ["year-month", "hour", "PM2.5"])

    def test_parts_named_in_other_cases_and_ways_are_combined(self):
        frame = self._hourly(year="YEAR", month="Month", day="DAY", hour="obs_hour")

        cleaned, _ = clean_dataframe(frame)

        self.assertEqual(cleaned["YEAR-Month-DAY-obs_hour"].iloc[3], pd.Timestamp("2017-01-01 00:00"))

    def test_minutes_and_seconds_join_the_timestamp(self):
        frame = self._hourly(year="Year", month="Month", day="Day", hour="Hour")
        frame["Minute"], frame["Second"] = 30, [0, 15, 30, 45, 0, 15]

        cleaned, _ = clean_dataframe(frame)

        self.assertEqual(
            cleaned["Year-Month-Day-Hour-Minute-Second"].iloc[1], pd.Timestamp("2016-12-31 22:30:15")
        )

    def test_a_day_or_hour_no_calendar_has_leaves_that_row_without_a_timestamp(self):
        frame = self._hourly()
        frame.loc[1, ["month", "day"]] = [2, 31]
        frame.loc[4, "hour"] = 24

        cleaned, report = clean_dataframe(frame)

        stamps = cleaned["year-month-day-hour"]
        self.assertEqual(stamps.isna().tolist(), [False, True, False, False, True, False])
        self.assertEqual(stamps.iloc[5], pd.Timestamp("2017-01-01 02:00"))
        self.assertEqual(report.unparsed_date_cells, 2)
        self.assertIn(
            "2 rows have no year-month-day-hour, because their year, month, day and hour make no valid "
            "date and time (such as 31 February or hour 24); the first is year 2016, month 2, day 31, "
            "hour 22.",
            report.notes,
        )

    def test_a_day_column_with_other_values_is_left_alone(self):
        frame = self._hourly()
        frame["day"] = ["Sat", "Sat", "Sat", "Sun", "Sun", "Sun"]

        cleaned, _ = clean_dataframe(frame)

        self.assertEqual(cleaned.columns.tolist(), ["year-month", "day", "hour", "PM2.5"])

    def test_a_month_column_with_other_values_is_left_alone(self):
        frame = pd.DataFrame({"Year": [2020, 2021], "Month": ["Spring", "Autumn"], "Yield": [1.0, 2.0]})

        cleaned, _ = clean_dataframe(frame)

        self.assertIn("Month", cleaned.columns)
        self.assertEqual(cleaned["Year"].iloc[1], pd.Timestamp("2021-01-01"))

    def test_years_outside_the_plausible_range_stay_numbers(self):
        frame = pd.DataFrame({"model_year": [70, 71, 82], "mpg": [18.0, 15.0, 31.0]})

        cleaned, report = clean_dataframe(frame)

        self.assertFalse(is_datetime64_any_dtype(cleaned["model_year"]))
        self.assertEqual(report.notes, ())

    def test_a_single_year_is_not_a_timeline(self):
        frame = pd.DataFrame({"Year": [2024, 2024, 2024], "Score": [1.0, 2.0, 3.0]})

        self.assertFalse(is_datetime64_any_dtype(clean_dataframe(frame)[0]["Year"]))

    def test_an_existing_date_column_takes_precedence(self):
        frame = pd.DataFrame(
            {"Date": ["2024-01-01", "2024-02-01"], "Year": [2024, 2024], "Sales": [1.0, 2.0]}
        )

        cleaned, _ = clean_dataframe(frame)

        self.assertTrue(is_datetime64_any_dtype(cleaned["Date"]))
        self.assertFalse(is_datetime64_any_dtype(cleaned["Year"]))

    def test_a_requested_date_column_of_years_is_converted_whatever_its_name(self):
        frame = pd.DataFrame({"Season": [1890, 1950, 2010], "Catch": [1.0, 2.0, 3.0]})

        self.assertFalse(is_datetime64_any_dtype(clean_dataframe(frame)[0]["Season"]))
        cleaned, report = clean_dataframe(frame, date="Season")
        self.assertEqual(cleaned["Season"].iloc[0], pd.Timestamp("1890-01-01"))
        self.assertTrue(any("Season holds whole years" in note for note in report.notes))

    def test_years_no_timestamp_can_hold_are_reported_as_unreadable(self):
        frame = pd.DataFrame({"Era": [1200, 1900, 2000], "Count": [1.0, 2.0, 3.0]})

        cleaned, report = clean_dataframe(frame, date="Era")

        self.assertTrue(pd.isna(cleaned["Era"].iloc[0]))
        self.assertEqual(report.unparsed_date_cells, 1)
        self.assertTrue(any("1 Era values could not be read" in note for note in report.notes))


class RepeatRowTests(unittest.TestCase):
    def _till(self):
        return pd.DataFrame(
            {
                "Date": ["2024-01-01"] * 4,
                "Product": ["Coffee"] * 4,
                "Revenue": [3.5, 3.5, 3.5, 3.5],
            }
        )

    def test_identical_rows_are_kept_and_the_total_survives(self):
        cleaned, report = clean_dataframe(self._till())

        self.assertEqual(len(cleaned), 4)
        self.assertAlmostEqual(cleaned["Revenue"].sum(), 14.0)
        self.assertEqual(report.duplicate_rows_removed, 0)
        self.assertEqual(report.duplicate_rows_found, 3)
        self.assertTrue(any("identical rows were kept" in note for note in report.notes))

    def test_drop_duplicates_removes_identical_rows(self):
        cleaned, report = clean_dataframe(self._till(), drop_duplicates=True)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(report.duplicate_rows_removed, 3)


class DateOrderingTests(unittest.TestCase):
    def test_a_day_over_twelve_settles_the_ordering(self):
        frame = pd.DataFrame({"Posting Date": [f"{d:02d}/03/2024" for d in range(1, 26)]})

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(cleaned["Posting Date"].min(), pd.Timestamp("2024-03-01"))
        self.assertEqual(cleaned["Posting Date"].max(), pd.Timestamp("2024-03-25"))
        self.assertTrue(any("day-first" in note for note in report.notes))

    def test_an_unsettleable_column_says_which_way_it_was_read(self):
        frame = pd.DataFrame({"Month": [f"01/{m:02d}/2024" for m in range(1, 13)]})

        _, report = clean_dataframe(frame)

        self.assertTrue(any("either way round" in note for note in report.notes))

    def test_iso_dates_are_never_called_ambiguous(self):
        frame = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=10).astype(str)})

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(cleaned["Date"].max(), pd.Timestamp("2024-01-10"))
        self.assertEqual(report.notes, ())


class TimezoneTests(unittest.TestCase):
    def test_an_offset_aware_column_is_analyzable(self):
        frame = pd.DataFrame(
            {
                "created_at": pd.date_range("2024-01-01", periods=30, tz="Asia/Kolkata"),
                "Revenue": range(30),
            }
        )

        cleaned, report = clean_dataframe(frame)

        self.assertFalse(isinstance(cleaned["created_at"].dtype, pd.DatetimeTZDtype))
        # The local wall-clock time is kept; values are not converted to UTC.
        self.assertEqual(cleaned["created_at"].iloc[0], pd.Timestamp("2024-01-01 00:00:00"))
        self.assertTrue(any("Timezone" in note for note in report.notes))

    def test_mixed_utc_offsets_still_become_a_date_column(self):
        frame = pd.DataFrame(
            {"created_at": ["2024-01-15T10:00:00+01:00", "2024-06-15T10:00:00+02:00"] * 10,
             "Revenue": range(20)}
        )

        cleaned, _ = clean_dataframe(frame)

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned["created_at"]))
        self.assertEqual(detect_roles(cleaned).date, "created_at")


class BusinessFormattedNumberTests(unittest.TestCase):
    def test_a_leading_minus_sign_is_never_lost(self):
        frame = pd.DataFrame({"Amount": ["-100", "1,000", "-$1,000.00", "$2,000.00", "(48.10)"]})

        cleaned, _ = clean_dataframe(frame)

        self.assertEqual(cleaned["Amount"].tolist(), [-100.0, 1000.0, -1000.0, 2000.0, -48.10])

    def test_european_decimals_keep_their_magnitude(self):
        frame = pd.DataFrame({"Amount": ["1.234,50", "2.345,60", "1 234,50", "1'234.50"]})

        cleaned, _ = clean_dataframe(frame)

        self.assertEqual(cleaned["Amount"].tolist(), [1234.5, 2345.6, 1234.5, 1234.5])

    def test_one_column_is_never_read_with_two_decimal_marks(self):
        frame = pd.DataFrame({
            "Error (mDa)": ["-0,307", "0,069", "-2,95", "-1,222"],
            "Mass": ["306,27883", "941,5104", "225,07509", "316,28438"],
        })

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(cleaned["Error (mDa)"].tolist(), [-0.307, 0.069, -2.95, -1.222])
        self.assertEqual(cleaned["Mass"].tolist(), [306.27883, 941.5104, 225.07509, 316.28438])
        named = "Commas mark the decimal point in Error (mDa), Mass"
        self.assertTrue(any(named in note for note in report.notes))

    def test_a_few_comma_decimals_do_not_turn_dot_decimals_into_thousands(self):
        values = ["5.66", "8.08", "6.207", "5.241", "4.87", "5.82", "5,62", "4.68"]
        frame = pd.DataFrame({"pH": values * 5})

        cleaned, _ = clean_dataframe(frame)

        self.assertIn(6.207, cleaned["pH"].tolist())
        self.assertIn(5.62, cleaned["pH"].tolist())
        self.assertLess(cleaned["pH"].max(), 14)

    def test_commas_grouping_thousands_still_read_as_thousands(self):
        frame = pd.DataFrame({"Amount": ["1,234", "5,678", "12", "99"]})

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(cleaned["Amount"].tolist(), [1234.0, 5678.0, 12.0, 99.0])
        self.assertFalse(any("decimal point" in note for note in report.notes))

    def test_a_value_plain_parsing_already_read_is_never_replaced(self):
        frame = pd.DataFrame({"Amount": ["1e3"] + ["1,000"] * 19})

        cleaned, _ = clean_dataframe(frame)

        self.assertEqual(int(cleaned["Amount"].isna().sum()), 0)
        self.assertEqual(cleaned["Amount"].iloc[0], 1000.0)

    def test_overflow_widening_sees_columns_that_inference_produced(self):
        frame = pd.DataFrame({"Amount": ["5000000000000000000"] * 2})

        cleaned, report = clean_dataframe(frame)

        self.assertGreater(float(cleaned["Amount"].sum()), 0)
        self.assertAlmostEqual(float(cleaned["Amount"].sum()), 1e19, delta=1e4)
        self.assertTrue(any("would not fit" in note for note in report.notes))

    def test_thousands_separators_do_not_delete_the_largest_values(self):
        frame = pd.DataFrame({"Revenue": ["950.00", "1,203.55", "12,400.10", "88.20"]})

        cleaned, _ = clean_dataframe(frame)

        self.assertAlmostEqual(cleaned["Revenue"].sum(), 14641.85)

    def test_currency_symbols_and_accounting_negatives_are_read(self):
        frame = pd.DataFrame({"Amount": ["$1,200.50", "$300.00", "(48.10)", "$0.00"]})

        cleaned, _ = clean_dataframe(frame)

        self.assertAlmostEqual(cleaned["Amount"].sum(), 1452.40)

    def test_a_slashed_date_is_never_read_as_a_number(self):
        frame = pd.DataFrame({"Month": [f"01/{m:02d}/2024" for m in range(1, 13)]})

        cleaned, _ = clean_dataframe(frame)

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned["Month"]))


class AccountingNegativeTests(unittest.TestCase):
    def test_currency_outside_the_parentheses(self):
        self.assertEqual(_read_formatted_number("$(100)"), -100.0)

    def test_sign_after_the_currency_symbol(self):
        self.assertEqual(_read_formatted_number("$-100"), -100.0)

    def test_parentheses_win_over_a_leading_plus(self):
        self.assertEqual(_read_formatted_number("+(100)"), -100.0)

    def test_markers_in_any_order_still_read_as_owed(self):
        for text in ("-$100", "($100)", "-(100)", "(-100)", "(1,234.50)"):
            with self.subTest(text=text):
                self.assertLess(_read_formatted_number(text), 0)

    def test_refund_column_keeps_its_values(self):
        column = pd.Series(["$1,000.00", "$(250.00)", "$-125.50", "$375.25"])
        read = _numeric_from_text(column)
        self.assertEqual(read.notna().sum(), 4)
        # 1,000.00 - 250.00 - 125.50 + 375.25
        self.assertAlmostEqual(float(read.sum()), 999.75)

    def test_a_doubled_marker_is_still_refused(self):
        for text in ("((100))", "--100", "$$100", "(100", "1,2,3", "12 34"):
            with self.subTest(text=text):
                self.assertIsNone(_read_formatted_number(text))

    def test_ordinary_numbers_are_unchanged(self):
        self.assertEqual(_read_formatted_number("100"), 100.0)
        self.assertEqual(_read_formatted_number("1,234"), 1234.0)
        self.assertEqual(_read_formatted_number("1234,50"), 1234.5)
        self.assertEqual(_read_formatted_number("1 234,50"), 1234.5)
        self.assertEqual(_read_formatted_number("12.5"), 12.5)


class NonFiniteValuesTests(unittest.TestCase):
    def test_an_infinity_is_excluded_from_the_total(self):
        frame = pd.DataFrame({
            "Date": pd.to_datetime(["2025-01-05", "2025-02-05", "2025-03-05"]),
            "Revenue": [1.0, np.inf, 3.0],
        })
        cleaned, _ = clean_dataframe(frame)
        self.assertEqual(float(cleaned["Revenue"].sum()), 4.0)
        self.assertTrue(pd.isna(cleaned["Revenue"].iloc[1]))

    def test_negative_infinity_too(self):
        frame = pd.DataFrame({"Revenue": [1.0, -np.inf, 3.0], "Team": list("abc")})
        cleaned, _ = clean_dataframe(frame)
        self.assertEqual(float(cleaned["Revenue"].sum()), 4.0)

    def test_the_text_inf_read_as_a_number_is_caught_too(self):
        """pandas parses the string "inf" into a float infinity."""
        frame = pd.DataFrame({"Revenue": ["1.0", "inf", "3.0"], "Team": list("abc")})
        cleaned, _ = clean_dataframe(frame)
        self.assertEqual(float(cleaned["Revenue"].sum()), 4.0)

    def test_non_finite_cells_are_reported(self):
        frame = pd.DataFrame({"Revenue": [1.0, np.inf, 3.0], "Team": list("abc")})
        _, report = clean_dataframe(frame)
        self.assertEqual(report.non_finite_cells, 1)
        self.assertTrue(any("Revenue" in note for note in report.notes), report.notes)

    def test_ordinary_numbers_are_untouched(self):
        frame = pd.DataFrame({"Revenue": [1.0, 2.0, 3.0], "Team": list("abc")})
        cleaned, report = clean_dataframe(frame)
        self.assertEqual(float(cleaned["Revenue"].sum()), 6.0)
        self.assertEqual(report.non_finite_cells, 0)


class UniqueColumnNameTests(unittest.TestCase):
    def test_a_name_the_suffix_would_collide_with_is_stepped_over(self):
        frame = pd.DataFrame([[1, 2, 3]], columns=["Amount", "Amount ", "Amount_2"])

        cleaned, _ = clean_dataframe(frame)

        self.assertEqual(len(set(cleaned.columns)), 3)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.dataframe = pd.DataFrame(
            {
                "sales": [10, 20, 30, 40, 1000],
                "units": [1, 2, 3, 4, 100],
                "region": ["West", "West", "West", "East", "West"],
            }
        )
        self.cleaned, self.report = clean_dataframe(self.dataframe)

    def test_profile_contains_quality_metrics(self):
        profile = column_profile(self.cleaned)
        self.assertEqual(profile["Column"].tolist(), ["sales", "units", "region"])
        self.assertIn("Missing %", profile.columns)

    def test_insights_are_computed_without_model_output(self):
        insights = generate_insights(self.cleaned)
        titles = {insight.title for insight in insights}
        self.assertIn("Strongest numeric relationship", titles)
        self.assertIn("Largest category share", titles)

    def test_markdown_report_states_local_processing(self):
        insights = generate_insights(self.cleaned)
        report = build_markdown_report(self.cleaned, self.report, insights, "Sales example")
        self.assertIn("# Automated Data Analysis Report", report)
        self.assertIn("No uploaded data was sent to an external AI service", report)


class KeywordScoreTests(unittest.TestCase):
    def test_the_last_word_decides_what_a_column_is(self):
        self.assertGreater(
            _keyword_score("Product Category", DIMENSION_KEYWORDS),
            _keyword_score("Product Container", DIMENSION_KEYWORDS),
        )
        self.assertGreater(
            _keyword_score("Region", DIMENSION_KEYWORDS),
            _keyword_score("Product Container", DIMENSION_KEYWORDS),
        )

    def test_a_trailing_qualifier_does_not_erase_the_keyword(self):
        self.assertGreater(_keyword_score("Revenue USD", MEASURE_KEYWORDS), 0)
        self.assertEqual(
            _keyword_score("Revenue", MEASURE_KEYWORDS),
            max(MEASURE_KEYWORDS["revenue"], 0),
        )

    def test_the_real_segment_is_chosen_over_a_packaging_attribute(self):
        frame = pd.DataFrame(
            {
                "Sales": np.arange(12, dtype=float),
                "Product Container": ["Small Box", "Jumbo Drum", "Large Box"] * 4,
                "Region": ["West", "East", "North", "South"] * 3,
                "Customer Segment": ["Consumer", "Corporate"] * 6,
            }
        )

        self.assertNotEqual(detect_roles(frame).dimension, "Product Container")

    def test_a_date_is_never_an_identifier(self):
        """"Order Date" contains an identifier word and every value is unique."""
        dates = pd.Series(pd.date_range("2023-01-01", periods=400))

        self.assertFalse(looks_like_identifier("Order Date", dates))
        self.assertTrue(looks_like_identifier("Order ID", pd.Series([f"A{n}" for n in range(400)])))

    def test_a_date_column_does_not_take_the_identifier_role(self):
        frame = pd.DataFrame(
            {
                "Order Date": pd.date_range("2023-01-01", periods=40),
                "Revenue": np.arange(40, dtype=float),
                "Channel": ["Direct", "Partner"] * 20,
            }
        )

        self.assertIsNone(detect_roles(frame).identifier)

    def test_total_reads_as_a_measure(self):
        self.assertGreater(_keyword_score("Total", MEASURE_KEYWORDS), 0)
        self.assertGreater(_keyword_score("Turnover", MEASURE_KEYWORDS), 0)
        self.assertGreater(
            _keyword_score("Total", MEASURE_KEYWORDS),
            _keyword_score("Passengers", MEASURE_KEYWORDS),
        )
        self.assertGreater(
            _keyword_score("Revenue", MEASURE_KEYWORDS),
            _keyword_score("Total", MEASURE_KEYWORDS),
        )


class RoleDeterminismTests(unittest.TestCase):
    def _hr_file(self, order):
        frame = pd.DataFrame(
            {
                "Employee ID": [f"E{index:03d}" for index in range(24)],
                "Postal Code": [10_000 + index * 37 for index in range(24)],
                "Annual Salary": [60_000.5 + index * 1_500 for index in range(24)],
                "Tenure Months": [index % 60 for index in range(24)],
                "Department": ["Eng", "Sales", "Ops"] * 8,
            }
        )
        return frame[order]

    def test_column_order_does_not_change_the_headline_metric(self):
        orders = (
            ["Employee ID", "Postal Code", "Annual Salary", "Tenure Months", "Department"],
            ["Employee ID", "Annual Salary", "Tenure Months", "Postal Code", "Department"],
            ["Tenure Months", "Employee ID", "Department", "Annual Salary", "Postal Code"],
        )

        detected = {detect_roles(self._hr_file(order)).measure for order in orders}

        self.assertEqual(len(detected), 1, f"measure depended on column order: {detected}")
        self.assertEqual(detected.pop(), "Annual Salary")

    def test_a_money_column_beside_an_invoice_is_not_a_row_identifier(self):
        frame = pd.DataFrame(
            {
                "Invoice Number": [f"INV-{index:05d}" for index in range(30)],
                "Invoice Amount": [1_000.0 + index * 13.5 for index in range(30)],
                "Days Overdue": list(range(30)),
                "Customer": ["A", "B", "C"] * 10,
            }
        )

        roles = detect_roles(frame)

        self.assertEqual(roles.measure, "Invoice Amount")
        self.assertEqual(roles.identifier, "Invoice Number")

    def test_a_parenthesised_unit_does_not_demote_the_head_noun(self):
        frame = pd.DataFrame(
            {
                "Revenue (USD)": [100.0 + index for index in range(20)],
                "Discount Amount": [1.0 + index for index in range(20)],
                "Channel": ["a", "b"] * 10,
            }
        )

        self.assertEqual(detect_roles(frame).measure, "Revenue (USD)")

    def test_camel_case_names_score_like_snake_case_ones(self):
        frame = pd.DataFrame(
            {
                "netRevenue": [100.0 + index for index in range(20)],
                "Discount Amount": [1.0 + index for index in range(20)],
                "Channel": ["a", "b"] * 10,
            }
        )

        self.assertEqual(detect_roles(frame).measure, "netRevenue")


class ExportedIndexTests(unittest.TestCase):
    def test_a_blank_row_does_not_save_the_row_numbers(self):
        frame = pd.DataFrame(
            {
                "Unnamed: 0": [0, 1, None, 3, 4, 5],
                "Region": ["N", "S", None, "N", "S", "N"],
                "Headcount": [10, 20, None, 30, 40, 50],
            }
        )

        cleaned, report = clean_dataframe(frame)

        self.assertNotIn("Unnamed: 0", cleaned.columns)
        self.assertEqual(report.index_columns_removed, 1)

    def test_a_named_counter_column_is_never_dropped(self):
        frame = pd.DataFrame({"Sequence": [0, 1, 2, 3], "Region": ["N", "S", "N", "S"]})

        cleaned, _ = clean_dataframe(frame)

        self.assertIn("Sequence", cleaned.columns)


class UnreadableDateColumnTests(unittest.TestCase):
    def test_a_column_that_failed_to_parse_as_a_date_is_not_a_segment(self):
        frame = pd.DataFrame(
            {
                "Date": [f"2024-06-{day:02d} maybe" for day in range(1, 13)],
                "Revenue": [100.0 + index for index in range(12)],
            }
        )

        roles = detect_roles(clean_dataframe(frame)[0])

        self.assertNotEqual(roles.dimension, "Date")


class MissingMarkerTests(unittest.TestCase):
    def test_missing_markers_in_a_numeric_column_become_missing(self):
        frame = pd.DataFrame({"OD600": ["0.81", "NA", "0.77", "nd", "<0.05", "0.92"]})
        cleaned, report = clean_dataframe(frame)
        self.assertEqual(cleaned["OD600"].dtype.kind, "f")
        self.assertEqual(int(cleaned["OD600"].isna().sum()), 3)
        self.assertTrue(any('"NA"' in note and "OD600" in note for note in report.notes), report.notes)

    def test_a_text_column_keeps_its_na_values(self):
        frame = pd.DataFrame({"Region": ["NA", "EU", "APAC", "NA"], "Revenue": [1.0, 2.0, 3.0, 4.0]})
        cleaned, _ = clean_dataframe(frame)
        self.assertEqual(cleaned["Region"].tolist(), ["NA", "EU", "APAC", "NA"])

    def test_a_column_of_only_markers_stays_text(self):
        frame = pd.DataFrame({"Note": ["nd", "nd", "NA"], "Value": [1.0, 2.0, 3.0]})
        cleaned, _ = clean_dataframe(frame)
        self.assertEqual(cleaned["Note"].dtype, object)

    def test_a_lone_repeated_token_is_a_marker_and_the_column_stays_numeric(self):
        frame = pd.DataFrame({"Precip": ["0", "T", "1.5", "T", "0.2", "0.8", "T", "2.1"] * 5})
        cleaned, report = clean_dataframe(frame)
        self.assertEqual(cleaned["Precip"].dtype.kind, "f")
        self.assertEqual(int(cleaned["Precip"].isna().sum()), 15)
        note = next(note for note in report.notes if "Precip" in note)
        self.assertIn('15 cells in Precip held "T"', note)
        self.assertIn("trace", note)

    def test_the_word_trace_reads_as_zero(self):
        frame = pd.DataFrame({"Snowfall": ["trace", "1.2", "0.4", "2.0", "0.0"] * 4})
        cleaned, report = clean_dataframe(frame)
        self.assertEqual(int((cleaned["Snowfall"] == 0).sum()), 8)
        self.assertTrue(any('held "trace" and were read as 0' in note for note in report.notes))

    def test_a_long_word_among_numbers_is_a_level_not_a_marker(self):
        frame = pd.DataFrame({"Dose": ["Control", "1", "5", "10"] * 5})
        cleaned, _ = clean_dataframe(frame)
        self.assertEqual(cleaned["Dose"].dtype, object)

    def test_values_below_a_comma_decimal_limit_are_censored_with_a_bias_note(self):
        frame = pd.DataFrame({"P mg/kg": ["<3,87", "5,10", "7,25", "<3,87", "9,00"] * 4})
        cleaned, report = clean_dataframe(frame)
        self.assertEqual(int(cleaned["P mg/kg"].isna().sum()), 8)
        note = next(note for note in report.notes if "P mg/kg" in note)
        self.assertIn("below a detection limit (<3.87)", note)
        self.assertIn("biased upward", note)

    def test_a_few_censored_values_get_no_bias_warning(self):
        values = ["<0.5"] + [f"{1 + i / 10:.1f}" for i in range(30)]
        cleaned, report = clean_dataframe(pd.DataFrame({"Zinc": values}))
        note = next(note for note in report.notes if "Zinc" in note)
        self.assertIn("below a detection limit (<0.5)", note)
        self.assertNotIn("biased", note)

    def test_cells_numbers_could_not_read_are_named(self):
        values = [f"{i / 10:.1f}" for i in range(40)] + ["1 1"]
        cleaned, report = clean_dataframe(pd.DataFrame({"Reading": values}))
        self.assertEqual(cleaned["Reading"].dtype.kind, "f")
        unread = [note for note in report.notes if "could not be read as a number" in note]
        self.assertTrue(unread and '"1 1"' in unread[0], report.notes)

    def test_not_stated_phrases_in_a_text_column_become_missing(self):
        frame = pd.DataFrame({
            "Drainage": ["Good", "No indica", "Poor", "NO INDICA", "ERROR: #N/A", "sin dato"],
            "pH": [5.1, 6.2, 4.8, 5.5, 6.0, 5.9],
        })
        cleaned, report = clean_dataframe(frame)
        self.assertEqual(int(cleaned["Drainage"].isna().sum()), 4)
        self.assertTrue(any("Drainage" in note and "not stated" in note for note in report.notes))

    def test_nd_stays_in_a_column_of_two_letter_codes(self):
        frame = pd.DataFrame({"State": ["ND", "SD", "MN", "ND", "IA"], "Sales": [1.0, 2.0, 3.0, 4.0, 5.0]})
        cleaned, _ = clean_dataframe(frame)
        self.assertEqual(cleaned["State"].tolist(), ["ND", "SD", "MN", "ND", "IA"])


class CleaningNoteTests(unittest.TestCase):
    def test_a_renamed_duplicate_header_is_noted_with_whether_it_differs(self):
        frame = pd.DataFrame({"diabetes": [1, 0, 1, 0], "diabetes.1": [1, 1, 1, 0], "age": [40, 50, 60, 70]})
        _, report = clean_dataframe(frame)
        self.assertIn(
            'The header "diabetes" appears more than once; the repeat is named "diabetes.1". '
            "They differ on 1 row.",
            report.notes,
        )

    def test_numbers_as_headers_are_not_a_repeat(self):
        frame = pd.DataFrame({"400": [0.1, 0.2], "400.5": [0.3, 0.4], "401": [0.5, 0.6]})
        _, report = clean_dataframe(frame)
        self.assertFalse(any("appears more than once" in note for note in report.notes))

    def test_a_duplicate_column_in_a_frame_is_noted(self):
        frame = pd.DataFrame([[1, 1, 5], [0, 0, 6]], columns=["flag", "flag", "x"])
        _, report = clean_dataframe(frame)
        expected = 'repeat is named "flag_2". They hold the same values.'
        self.assertTrue(any(expected in note for note in report.notes), report.notes)

    def test_dropped_empty_columns_are_named(self):
        frame = pd.DataFrame({"FT": [None, None], "Value": [1.0, 2.0], "WTE": ["", " "]})
        cleaned, report = clean_dataframe(frame)
        self.assertEqual(list(cleaned.columns), ["Value"])
        self.assertIn("2 empty columns were dropped: FT and WTE.", report.notes)


class IdentifierNameCleaningTests(unittest.TestCase):
    def test_words_containing_id_are_still_parsed_as_numbers(self):
        frame = pd.DataFrame({"Humidity": ["45.1", "NA", "47.3"], "Width": ["1,204", "980", "1,010"]})
        cleaned, _ = clean_dataframe(frame)
        self.assertEqual(cleaned["Humidity"].dtype.kind, "f")
        self.assertEqual(cleaned["Width"].tolist(), [1204.0, 980.0, 1010.0])

    def test_key_columns_keep_their_leading_zeros(self):
        frame = pd.DataFrame({"Account Number": ["00042", "00017"], "Amount": [1.0, 2.0]})
        cleaned, _ = clean_dataframe(frame)
        self.assertEqual(cleaned["Account Number"].tolist(), ["00042", "00017"])

    def test_whole_numbers_under_a_key_name_stay_text(self):
        cleaned, _ = clean_dataframe(pd.DataFrame({"Order No": ["1001", "1002", "1003"]}))
        self.assertEqual(cleaned["Order No"].tolist(), ["1001", "1002", "1003"])

    def test_decimals_under_a_key_name_are_read_as_numbers(self):
        frame = pd.DataFrame({"NO": ["36.57", "12.10", "NA"], "Sample Code": ["1.5", "2.25", "3.0"]})
        cleaned, _ = clean_dataframe(frame)
        self.assertEqual(cleaned["NO"].dtype.kind, "f")
        self.assertEqual(cleaned["Sample Code"].tolist(), [1.5, 2.25, 3.0])


class LabelVariantTests(unittest.TestCase):
    def test_labels_differing_only_in_case_accents_or_spacing_are_merged(self):
        frame = pd.DataFrame({
            "Tiempo": ["MAS DE 5"] * 4 + ["Mas de 5", "mas De 5", "de 1 a 5 años", "De 1 a 5  anos"],
            "MO": [1.0] * 8,
        })

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(
            cleaned["Tiempo"].value_counts().to_dict(), {"MAS DE 5": 6, "de 1 a 5 años": 2}
        )
        note = next(note for note in report.notes if note.startswith("3 Tiempo values"))
        self.assertIn('"Mas de 5" and "mas De 5" became "MAS DE 5"', note)
        self.assertIn("1 other label likewise", note)

    def test_distinct_labels_and_keys_are_left_alone(self):
        frame = pd.DataFrame({"Region": ["North", "South"] * 3, "Account ID": ["ab12", "AB12", "cd34"] * 2})

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(sorted(cleaned["Account ID"].unique()), ["AB12", "ab12", "cd34"])
        self.assertFalse(any("spelled differently" in note for note in report.notes))


class LeadingZeroTests(unittest.TestCase):
    def test_codes_with_leading_zeros_stay_text(self):
        frame = pd.DataFrame({
            "Zipcode": ["02134", "02139", "10001", "-", "02115"],
            "Store": ["007", "012", "100", "101", "102"],
            "Phone No.": ["0123456789"] * 5,
        })
        cleaned, _ = clean_dataframe(frame)
        self.assertEqual(cleaned["Zipcode"].tolist(), ["02134", "02139", "10001", "-", "02115"])
        self.assertEqual(cleaned["Store"].tolist(), ["007", "012", "100", "101", "102"])
        self.assertEqual(cleaned["Phone No."].iloc[0], "0123456789")

    def test_decimals_below_one_are_still_numbers(self):
        cleaned, _ = clean_dataframe(pd.DataFrame({"OD600": ["0.81", "0.05", "NA", "1.20"]}))
        self.assertEqual(cleaned["OD600"].dtype.kind, "f")


class ClockTimeTests(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2024-01-01 15:06:28", periods=120, freq="s").strftime("%H:%M:%S")
        self.frame = pd.DataFrame({"Time": times, "Signal": np.arange(120.0)})

    def test_times_of_day_are_never_given_a_date(self):
        cleaned, report = clean_dataframe(self.frame)

        times = cleaned["Time"]
        self.assertTrue(is_datetime64_any_dtype(times))
        self.assertTrue(times.map(is_undated).all())
        self.assertEqual(times.iloc[0].strftime("%H:%M:%S"), "15:06:28")
        self.assertNotEqual(times.iloc[0].normalize(), pd.Timestamp.today().normalize())
        note = next(note for note in report.notes if note.startswith("Time holds"))
        self.assertIn("no calendar date (15:06:28 to 15:08:27)", note)

    def test_a_log_that_passes_midnight_moves_on_to_the_next_day(self):
        frame = pd.DataFrame(
            {"Clock": ["23:59:58", "23:59:59", "00:00:00", "00:00:01"], "Value": [1, 2, 3, 4]}
        )

        cleaned, report = clean_dataframe(frame)

        self.assertTrue(cleaned["Clock"].is_monotonic_increasing)
        self.assertEqual(format_moment(cleaned["Clock"].iloc[-1]), "day 2 00:00:01")
        self.assertTrue(any("passing midnight 1 time" in note for note in report.notes))

    def test_a_bare_time_in_a_column_of_dates_is_not_read_as_today(self):
        frame = pd.DataFrame({"Date": [
            "2024-03-01 10:00", "2024-03-02 11:00", "2024-03-03 12:00", "2024-03-04 13:00", "15:47",
        ]})

        cleaned, report = clean_dataframe(frame)

        self.assertTrue(pd.isna(cleaned["Date"].iloc[-1]))
        self.assertEqual(report.unparsed_date_cells, 1)

    def test_a_date_column_and_a_time_column_combine_into_one(self):
        frame = pd.DataFrame({
            "Date": ["2024-03-01", "2024-03-01", "2024-03-02"],
            "Time": ["23:00:00", "23:30:00", "00:15:00"],
            "Reading": [1.0, 2.0, 3.0],
        })

        cleaned, report = clean_dataframe(frame)

        self.assertNotIn("Time", cleaned.columns)
        expected = pd.to_datetime(["2024-03-01 23:00", "2024-03-01 23:30", "2024-03-02 00:15"])
        self.assertEqual(cleaned["Date"].tolist(), expected.tolist())
        self.assertTrue(any("combined into one column, Date" in note for note in report.notes))


class ElapsedTimeTests(unittest.TestCase):
    def test_elapsed_seconds_become_the_time_axis(self):
        frame = pd.DataFrame(
            {"Time (s)": [0.0, 0.5, 1.0, 1.5, 2.0], "Absorbance": [0.1, 0.2, 0.3, 0.4, 0.5]}
        )

        cleaned, report = clean_dataframe(frame)

        times = cleaned["Time (s)"]
        self.assertTrue(is_datetime64_any_dtype(times))
        self.assertEqual(times.iloc[-1] - times.iloc[0], pd.Timedelta(seconds=2))
        self.assertTrue(times.map(is_undated).all())
        self.assertTrue(any("counts seconds from the start" in note for note in report.notes))
        self.assertEqual(detect_roles(cleaned).date, "Time (s)")

    def test_minutes_are_counted_in_minutes(self):
        frame = pd.DataFrame({"elapsed_min": [0, 5, 10, 15], "OD600": [0.1, 0.2, 0.4, 0.8]})

        cleaned, _ = clean_dataframe(frame)

        self.assertEqual(cleaned["elapsed_min"].iloc[-1].strftime("%H:%M"), "00:15")

    def test_a_duration_measured_per_row_is_not_a_time_axis(self):
        frame = pd.DataFrame({
            "Time to failure (h)": [30.0, 12.0, 45.0, 8.0],
            "Time (s)": [3.0, 1.0, 2.0, 5.0],
        })

        cleaned, _ = clean_dataframe(frame)

        self.assertFalse(is_datetime64_any_dtype(cleaned["Time to failure (h)"]))
        # Not in time order, so a per-row duration rather than a log.
        self.assertFalse(is_datetime64_any_dtype(cleaned["Time (s)"]))

    def test_a_requested_elapsed_column_without_a_unit_counts_seconds(self):
        frame = pd.DataFrame({"t": [3, 1, 2], "Signal": [1.0, 2.0, 3.0]})

        cleaned, _ = clean_dataframe(frame, date="t")

        self.assertEqual(cleaned["t"].iloc[0].strftime("%H:%M:%S"), "00:00:03")


class MixedDateFormatTests(unittest.TestCase):
    def test_iso_dates_with_and_without_a_time_both_parse(self):
        frame = pd.DataFrame(
            {"Datetime": ["2015-01-01 01:00:00", "2015-01-01 02:00:00", "2019-04-02", "2019-04-03"]}
        )

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(cleaned["Datetime"].notna().sum(), 4)
        self.assertEqual(report.unparsed_date_cells, 0)


class PerValueCleaningTests(unittest.TestCase):
    def test_whole_numbers_held_as_text_stay_whole(self):
        frame = pd.DataFrame({"Count": ["1", "2", "3", "2"], "Site": ["a", "b", "a", "b"]})

        cleaned, _ = clean_dataframe(frame)

        self.assertEqual(str(cleaned["Count"].dtype), "int64")

    def test_repeated_values_are_trimmed_and_parsed_alike(self):
        frame = pd.DataFrame({
            "Amount": [" $1,200.50 ", "$1,200.50", None, "(30)"] * 50,
            "Store": ["North", "South"] * 100,
        })

        cleaned, _ = clean_dataframe(frame)

        self.assertEqual(cleaned["Amount"].iloc[:2].tolist(), [1200.5, 1200.5])
        self.assertTrue(pd.isna(cleaned["Amount"].iloc[2]))
        self.assertEqual(cleaned["Amount"].iloc[3], -30.0)

    def test_a_large_table_of_text_columns_is_cleaned_quickly(self):
        rng = np.random.default_rng(31)
        rows = 250_000
        data = {f"label_{i}": rng.choice([f"L{j}" for j in range(300)], rows) for i in range(10)}
        for i in range(3):
            data[f"reading_{i}"] = rng.normal(50, 10, rows).round(2).astype(str)
        frame = pd.DataFrame(data)

        started = time.perf_counter()
        cleaned, _ = clean_dataframe(frame)
        elapsed = time.perf_counter() - started

        self.assertEqual(cleaned["reading_0"].dtype.kind, "f")
        # About half a second on a laptop; the margin absorbs slow CI machines.
        self.assertLess(elapsed, 4.0)


class MissingValueCodeTests(unittest.TestCase):
    def test_a_code_is_set_to_missing_in_every_numeric_column_with_a_note_each(self):
        rng = np.random.default_rng(32)
        frame = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=200).strftime("%Y-%m-%d"),
            "PM": np.r_[rng.normal(955, 5, 150), [-999.0] * 50],
            "TMK": np.r_[rng.normal(10, 6, 180), [-999.0] * 20],
        })

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(cleaned[["PM", "TMK"]].isna().sum().tolist(), [50, 20])
        self.assertAlmostEqual(cleaned["PM"].mean(), 955, delta=3)
        self.assertIn("50 cells in PM held -999", " ".join(report.notes))
        self.assertIn("20 cells in TMK held -999", " ".join(report.notes))
        self.assertEqual([issue.columns for issue in report.blanked_codes], [("PM", "TMK")])

    def test_a_table_without_codes_is_unchanged(self):
        frame = pd.DataFrame({"Price": [9_999.0, 120.0, 3_000.0, 450.0] * 5})

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(int(cleaned["Price"].isna().sum()), 0)
        self.assertEqual(report.blanked_codes, ())


class TwoRowHeaderTests(unittest.TestCase):
    def test_group_labels_over_the_names_make_a_two_row_header(self):
        frame = pd.DataFrame({
            "Weld No.": [None, "L-a", "L-b", "L-c"],
            "Inputs": ["Pin Dia (mm)", 5, 7, 7],
            "Unnamed: 2": ["Speed (rpm)", 710, 710, 450],
            "Outputs": ["UTS (MPa)", 200.5, 205.2, 196.1],
            "Unnamed: 4": ["% Elongation", 11.3, 15.3, 8.4],
        })

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(
            list(cleaned.columns), ["Weld No.", "Pin Dia (mm)", "Speed (rpm)", "UTS (MPa)", "% Elongation"]
        )
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(cleaned["UTS (MPa)"].dtype.kind, "f")
        self.assertFalse(cleaned.isna().any().any())
        note = next(note for note in report.notes if note.startswith("The header has two rows"))
        self.assertIn('"Inputs" and "Outputs" label groups of columns', note)

    def test_a_label_over_one_column_is_joined_with_the_text_below_it(self):
        frame = pd.DataFrame({
            "Site": [None, "A", "B", "C"],
            "Temperature": ["(°C)", 20.1, 21.3, 19.8],
            "Readings": ["pH", 6.1, 6.4, 7.0],
            "Unnamed: 3": ["Nitrate (mg/L)", 1.2, 0.8, 2.5],
        })

        cleaned, _ = clean_dataframe(frame)

        self.assertEqual(list(cleaned.columns), ["Site", "Temperature (°C)", "pH", "Nitrate (mg/L)"])

    def test_a_first_row_of_numbers_is_data(self):
        frame = pd.DataFrame({
            "Unnamed: 0": ["x", "y", "z", "w"], "Value": [1.5, 2.5, 3.5, 4.5], "Count": [1, 2, 3, 4],
        })

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(len(cleaned), 4)
        self.assertIn("Value", cleaned.columns)
        self.assertFalse(any("two rows" in note for note in report.notes))


class WholeNumberDateTests(unittest.TestCase):
    def test_yyyymmdd_numbers_become_dates_when_the_table_has_no_other(self):
        days = pd.date_range("1954-06-01", periods=40)
        frame = pd.DataFrame({"MESS_DATUM": days.strftime("%Y%m%d").astype(int), "TMK": np.arange(40.0)})

        cleaned, report = clean_dataframe(frame)

        self.assertTrue(is_datetime64_any_dtype(cleaned["MESS_DATUM"]))
        self.assertEqual(cleaned["MESS_DATUM"].tolist(), list(days))
        self.assertIn(
            "MESS_DATUM holds whole numbers written as YYYYMMDD (01 Jun 1954 to 10 Jul 1954), so it was "
            "read as dates.",
            report.notes,
        )

    def test_yyyymm_numbers_under_a_date_name_become_months_beside_another_date(self):
        frame = pd.DataFrame({
            "Run Date": ["2024-01-05", "2024-02-07", "2024-03-04"],
            "Period": [202401, 202402, 202403],
            "Yield": [1.0, 2.0, 3.0],
        })

        cleaned, report = clean_dataframe(frame)

        months = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])
        self.assertEqual(cleaned["Period"].tolist(), list(months))
        self.assertTrue(any("YYYYMM (Jan 2024 to Mar 2024)" in note for note in report.notes))

    def test_numbers_that_are_not_all_valid_dates_stay_numbers(self):
        frame = pd.DataFrame({"Batch": [20240101, 20241301, 20240102], "Yield": [1.0, 2.0, 3.0]})

        cleaned, _ = clean_dataframe(frame)

        self.assertEqual(cleaned["Batch"].dtype.kind, "i")

    def test_numbers_that_look_like_dates_stay_numbers_beside_a_real_date(self):
        frame = pd.DataFrame({
            "Date": ["2024-01-05", "2024-02-07", "2024-03-04"],
            "Reads": [20240101, 20240102, 20240103],
        })

        cleaned, _ = clean_dataframe(frame)

        self.assertEqual(cleaned["Reads"].dtype.kind, "i")

    def test_a_column_named_as_the_date_is_read_whatever_else_the_table_holds(self):
        frame = pd.DataFrame({
            "Date": ["2024-01-05", "2024-02-07", "2024-03-04"],
            "Reads": [20240101, 20240102, 20240103],
        })

        cleaned, _ = clean_dataframe(frame, date="Reads")

        self.assertTrue(is_datetime64_any_dtype(cleaned["Reads"]))


class NotDetectedTests(unittest.TestCase):
    def test_nd_in_a_numeric_column_is_below_a_detection_limit(self):
        frame = pd.DataFrame({"Lead": ["0.81", "ND", "0.77", "0.92", "ND", "1.10"]})

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(int(cleaned["Lead"].isna().sum()), 2)
        note = next(note for note in report.notes if "Lead" in note)
        self.assertIn("2 cells in Lead (33.3%) were below a detection limit", note)

    def test_a_limit_equal_to_another_column_on_its_row_is_named(self):
        depth = [1.5, 2.0, 3.5, 4.0, 2.5, 6.0, 3.0, 5.5]
        clarity = [">1.5", ">2", "1.2", ">4", ">2.5", "2.2", ">3", ">5.5"]
        frame = pd.DataFrame({"Depth (m)": depth * 3, "Clarity (m)": clarity * 3})

        _, report = clean_dataframe(frame)

        note = next(note for note in report.notes if note.startswith("On "))
        self.assertIn('On 18 of the 18 rows where Clarity (m) is written as ">X"', note)
        self.assertIn("the reading reached Depth (m)", note)

    def test_a_shared_upper_limit_is_not_matched_to_a_column(self):
        frame = pd.DataFrame({"Steps": [100.0] * 6 + [40.0, 60.0], "Count": [">100"] * 6 + ["40", "60"]})

        _, report = clean_dataframe(frame)

        self.assertFalse([note for note in report.notes if "reached" in note])

    def test_nd_beside_values_written_with_a_limit_is_not_determined(self):
        frame = pd.DataFrame({"Zinc": ["<0.4", "ND", "1.2", "0.9", "ND", "2.2"] * 3})

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(int(cleaned["Zinc"].isna().sum()), 9)
        notes = " ".join(note for note in report.notes if "Zinc" in note)
        self.assertIn('6 cells in Zinc read as missing: "ND".', notes)
        self.assertIn("not determined", notes)
        self.assertIn("3 cells in Zinc (16.7%) were below a detection limit (<0.4)", notes)

    def test_nd_among_readings_written_as_text_is_not_detected(self):
        frame = pd.DataFrame({
            "Particles": ["5.66 ± 1.69 items/g", "ND", "14 items/L", "0.179 items/g", "ND"],
            "Food": ["Fish", "Salt", "Water", "Fish", "Salt"],
        })

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(int(cleaned["Particles"].isna().sum()), 2)
        notes = " ".join(note for note in report.notes if "Particles" in note)
        self.assertIn("2 cells in Particles (40.0%) were below a detection limit", notes)
        self.assertNotIn("not stated", notes)

    def test_readings_written_with_a_spread_and_a_unit_are_read_as_their_mean(self):
        frame = pd.DataFrame({
            "particle_number": ["5.66 ± 1.69 items/g", "0.43 items/ g", "2.1 items/g", "0.87 – 7.86 items/g"],
        })

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(cleaned["particle_number"].tolist()[:3], [5.66, 0.43, 2.1])
        self.assertTrue(np.isnan(cleaned["particle_number"].iloc[3]))
        self.assertNotIn("particle_number unit", cleaned.columns)
        self.assertTrue(any("Every one is in items/g" in note for note in report.notes))

    def test_readings_in_several_units_get_a_unit_column(self):
        frame = pd.DataFrame({"Level": ["5 mg/L", "7 mg/L", "0.2 g/L", "0.3 g/L", "4.5 *10^(6) mg/L"]})

        cleaned, _ = clean_dataframe(frame)

        self.assertEqual(cleaned["Level unit"].tolist()[:4], ["mg/L", "mg/L", "g/L", "g/L"])
        self.assertTrue(np.isnan(cleaned["Level"].iloc[4]))

    def test_yes_and_no_words_in_a_0_1_column_read_as_1_and_0(self):
        frame = pd.DataFrame({"liver_function_impaired": ["0", "1", "0", "true", "0", "False", "1", "0"]})

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(cleaned["liver_function_impaired"].tolist(), [0, 1, 0, 1, 0, 0, 1, 0])
        self.assertTrue(any('held "true", "False"' in note for note in report.notes))


class RespeltLabelTests(unittest.TestCase):
    def test_a_letter_and_the_one_word_it_starts_are_one_label(self):
        frame = pd.DataFrame({"gender": ["M"] * 50 + ["F"] * 40 + ["male"] * 6 + ["female"] * 5})

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(cleaned["gender"].value_counts().to_dict(), {"male": 56, "female": 45})
        self.assertTrue(any('"M" into "male"' in note for note in report.notes))

    def test_a_rare_spelling_one_letter_off_joins_the_common_one(self):
        frame = pd.DataFrame({"ethnicity": ["Hispanic"] * 30 + ["Hipanic"] * 4 + ["Asian"] * 40})

        cleaned, _ = clean_dataframe(frame)

        self.assertEqual(cleaned["ethnicity"].value_counts().to_dict(), {"Asian": 40, "Hispanic": 34})

    def test_codes_and_levels_that_differ_on_purpose_stay_apart(self):
        frame = pd.DataFrame({
            "blood": ["A", "B", "AB", "O"] * 25,
            "arm": ["Type A"] * 90 + ["Type B"] * 10,
            "size": ["Large"] * 90 + ["Larger"] * 10,
        })

        cleaned, _ = clean_dataframe(frame)

        self.assertEqual(cleaned["blood"].nunique(), 4)
        self.assertEqual(cleaned["arm"].nunique(), 2)


class OneHotTests(unittest.TestCase):
    def test_columns_marking_one_level_each_are_joined_into_one_category(self):
        frame = pd.DataFrame({
            "Rainfall": np.arange(100.0),
            "Soil_Type_Gravel": [1, 0, 0, 0] * 25,
            "Soil_Type_Sand": [0, 1, 0, 0] * 25,
            "Soil_Type_Silt": [0, 0, 1, 0] * 25,
        })

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(cleaned.columns.tolist(), ["Rainfall", "Soil_Type"])
        levels = {"Gravel": 25, "Sand": 25, "Silt": 25, "other": 25}
        self.assertEqual(cleaned["Soil_Type"].value_counts().to_dict(), levels)
        self.assertTrue(any("joined into one column, Soil_Type" in note for note in report.notes))

    def test_separate_yes_no_questions_are_left_alone(self):
        frame = pd.DataFrame({
            "has_pool": [1, 0, 0, 0] * 25,
            "has_garage": [0, 1, 0, 0] * 25,
            "has_garden": [0, 0, 1, 0] * 25,
            "Smoker_current": [1, 0, 1, 0] * 25,
            "Smoker_former": [0, 1, 1, 0] * 25,
        })

        cleaned, _ = clean_dataframe(frame)

        self.assertEqual(cleaned.columns.tolist(), frame.columns.tolist())


class PlaceholderTests(unittest.TestCase):
    def test_no_data_phrases_in_other_languages_become_missing(self):
        frame = pd.DataFrame({
            "Category": ["BAIK", "TIDAK ADA DATA", "SEDANG", "keine Daten", "sem dados", "BAIK"],
            "Index": [40.0, 0.0, 70.0, 0.0, 0.0, 45.0],
        })

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(cleaned["Category"].dropna().tolist(), ["BAIK", "SEDANG", "BAIK"])
        self.assertTrue(any("not stated" in note and "Category" in note for note in report.notes))

    def test_placeholder_labels_become_missing_with_a_note(self):
        frame = pd.DataFrame({
            "Cancer Type": ["BRCA", "UNABLE TO CLASSIFY", "LUAD", "Unknown", "unclassified", "BRCA"],
            "Z": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        })

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(cleaned["Cancer Type"].dropna().tolist(), ["BRCA", "LUAD", "BRCA"])
        note = next(note for note in report.notes if "Cancer Type" in note)
        self.assertIn("3 cells in Cancer Type read as missing", note)
        self.assertIn("not a group of its own", note)


class MangledRangeTests(unittest.TestCase):
    def test_a_range_a_spreadsheet_turned_into_a_date_is_restored(self):
        frame = pd.DataFrame({"Depth": ["0-10", "Oct-30", "30-60", "60-90"] * 3, "Ca": np.arange(12.0)})

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(cleaned["Depth"].unique().tolist(), ["0-10", "10-30", "30-60", "60-90"])
        self.assertIn(
            '3 Depth values read "Oct-30", which is how a spreadsheet shows the range "10-30" once it '
            'has read it as a date; they were restored to "10-30".',
            report.notes,
        )

    def test_a_range_that_fits_no_other_is_only_named(self):
        frame = pd.DataFrame({"Band": ["0-5", "5-15", "Mar-40", "15-25"] * 2, "Ca": np.arange(8.0)})

        cleaned, report = clean_dataframe(frame)

        self.assertIn("Mar-40", cleaned["Band"].tolist())
        named = 'may be how a spreadsheet shows the range "3-40"'
        self.assertTrue(any(named in note for note in report.notes))


def hourly_log(start, hours, value=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Datetime": pd.date_range(start, periods=hours, freq="h"),
        "PM2.5": value + rng.random(hours),
    })


def hourly_then_daily():
    hourly = hourly_log("2024-01-01 01:00", 24 * 30)
    daily = pd.DataFrame({"Datetime": pd.date_range("2024-01-01", periods=30, freq="D"), "PM2.5": 1.0})
    return pd.concat([hourly, daily], ignore_index=True)


class CoarserRowsTests(unittest.TestCase):
    def test_daily_summaries_after_hourly_rows_are_set_aside(self):
        kept, note = set_aside_coarser_rows(hourly_then_daily(), "Datetime")

        self.assertEqual(len(kept), 24 * 30)
        self.assertIn("The last 30 rows give Datetime as a date with no time of day", note)

    def test_a_reading_at_midnight_stays_with_the_hourly_rows(self):
        kept, _ = set_aside_coarser_rows(hourly_then_daily(), "Datetime")

        self.assertEqual(kept["Datetime"].iloc[-1], pd.Timestamp("2024-01-31 00:00"))

    def test_an_hourly_log_alone_is_untouched(self):
        frame = hourly_log("2024-01-01", 24 * 30)

        kept, note = set_aside_coarser_rows(frame, "Datetime")

        self.assertEqual(len(kept), len(frame))
        self.assertEqual(note, "")


class StackedLogTests(unittest.TestCase):
    def test_a_log_that_starts_again_gets_a_block_column(self):
        frame = pd.concat(
            [hourly_log("2024-01-01", 500, 1.0), hourly_log("2024-01-01", 500, 2.0)],
            ignore_index=True,
        )

        labelled, block, note = separate_stacked_logs(frame, "Datetime")

        self.assertEqual(block, "Block")
        self.assertEqual(
            labelled["Block"].iloc[[0, 499, 500, 999]].tolist(),
            ["Block 1", "Block 1", "Block 2", "Block 2"],
        )
        self.assertIn("holds 2 logs one after another", note)
        self.assertIn("500 timestamps appear in more than one of them", note)
        self.assertEqual(detect_roles(labelled).dimension, "Block")

    def test_a_file_sorted_by_site_is_already_labelled(self):
        frame = pd.concat(
            [hourly_log("2024-01-01", 500).assign(Site=site) for site in ("North", "South", "East")],
            ignore_index=True,
        )

        labelled, block, note = separate_stacked_logs(frame, "Datetime")

        self.assertIsNone(block)
        self.assertNotIn("Block", labelled.columns)
        self.assertEqual(note, "")

    def test_a_short_step_back_is_not_a_new_log(self):
        frame = hourly_log("2024-01-01", 1000)
        frame.loc[600, "Datetime"] = frame.loc[598, "Datetime"]

        _, block, note = separate_stacked_logs(frame, "Datetime")

        self.assertIsNone(block)
        self.assertEqual(note, "")

    def test_a_log_with_many_small_steps_back_is_one_log(self):
        frame = hourly_log("2020-01-01", 50_000)
        # A clock nudged back by minutes every 40 rows.
        frame.loc[::40, "Datetime"] -= pd.Timedelta(minutes=90)

        _, block, note = separate_stacked_logs(frame, "Datetime")

        self.assertIsNone(block)
        self.assertEqual(note, "")

    def test_readings_recorded_side_by_side_get_a_note(self):
        stamps = pd.date_range("2024-01-01", periods=500, freq="min")
        frame = pd.DataFrame({
            "ts": np.repeat(stamps, 2),
            "CO2": np.tile([400.0, 900.0], 500) + np.arange(1000) % 7,
        })

        _, block, note = separate_stacked_logs(frame, "ts")

        self.assertIsNone(block)
        self.assertIn("appear 2 times with different readings", note)


if __name__ == "__main__":
    unittest.main()