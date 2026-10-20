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
)
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

    def test_bare_years_stay_numbers(self):
        frame = pd.DataFrame({"Year": ["1949", "1950", "1951"], "Passengers": [1, 2, 3]})

        cleaned, _ = clean_dataframe(frame)

        self.assertFalse(is_datetime64_any_dtype(cleaned["Year"]))

    def test_text_categories_are_not_mistaken_for_dates(self):
        frame = pd.DataFrame(
            {"Container": ["Small Box", "Jumbo Drum", "Large Box"], "Sales": [1.0, 2.0, 3.0]}
        )

        cleaned, _ = clean_dataframe(frame)

        self.assertFalse(is_datetime64_any_dtype(cleaned["Container"]))


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


if __name__ == "__main__":
    unittest.main()