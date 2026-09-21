"""The ingestion contract: what leaves the reader is what the cleaner keeps.

Each test here pins a finding from an independent review of the read -> clean
-> detect path. They are written against the reviewer's own fixtures, and each
one failed before the fix it names.
"""

import io
import unittest
import zipfile

import numpy as np
import openpyxl
import pandas as pd

from analysis import OFFSETS_DROPPED_NOTE, _read_formatted_number, clean_dataframe
from file_io import list_excel_sheets, read_tabular_file
from schema import detect_roles, is_identifier_name


def workbook_bytes(rows: list[list[object]]) -> bytes:
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Data"
    for row in rows:
        sheet.append(row)
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def truncated_member(contents: bytes, member: str, cut: int) -> bytes:
    """The same workbook with the tail of one XML part missing."""
    source = zipfile.ZipFile(io.BytesIO(contents))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == member:
                data = data[: len(data) - cut]
            archive.writestr(item, data)
    return out.getvalue()


class IdentifierNameTests(unittest.TestCase):
    """One whole-word test decides what is a key, for the reader and the cleaner."""

    def test_the_last_word_or_the_whole_name_says_key(self):
        names = ("id", "Customer ID", "Account Number", "SKU", "Order No", "ZIP", "order_no", "customerId")
        for name in names:
            with self.subTest(name=name):
                self.assertTrue(is_identifier_name(name))

    def test_a_substring_is_not_a_word(self):
        for name in ("Paid Amount", "Grid Value", "Valid Amount", "Revenue", "Skunk Count", "Zipper Sales"):
            with self.subTest(name=name):
                self.assertFalse(is_identifier_name(name))

    def test_identifiers_survive_reading_and_cleaning_end_to_end(self):
        raw = (
            b"Customer ID,Account Number,SKU,Order No,ZIP,Revenue\n"
            b"00042,00042,00042,00042,00042,10\n"
            b"00007,00007,00007,00007,00007,20\n"
        )

        cleaned, _ = clean_dataframe(read_tabular_file(raw, "orders.csv"))

        for column in ("Customer ID", "Account Number", "SKU", "Order No", "ZIP"):
            with self.subTest(column=column):
                self.assertEqual(cleaned[column].tolist(), ["00042", "00007"])
        self.assertEqual(cleaned["Revenue"].tolist(), [10, 20])

    def test_names_that_merely_contain_id_are_still_read_as_numbers(self):
        frame = pd.DataFrame(
            {
                "Paid Amount": ["$1,000", "$2,000"],
                "Grid Value": ["$1,000", "$3,000"],
                "Valid Amount": ["$1,000", "$4,000"],
            }
        )

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(cleaned["Paid Amount"].tolist(), [1000.0, 2000.0])
        self.assertEqual(cleaned["Grid Value"].tolist(), [1000.0, 3000.0])
        self.assertEqual(cleaned["Valid Amount"].tolist(), [1000.0, 4000.0])
        self.assertEqual(report.numeric_columns_inferred, 3)

    def test_an_identifier_column_of_dates_is_neither_a_date_nor_the_date_role(self):
        frame = pd.DataFrame({"Customer ID": ["2024-01-01", "2024-01-02"] * 5, "Revenue": range(10)})

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(cleaned["Customer ID"].tolist(), ["2024-01-01", "2024-01-02"] * 5)
        self.assertEqual(report.datetime_columns_inferred, 0)
        self.assertIsNone(detect_roles(cleaned).date)


class FormattedNumberTests(unittest.TestCase):
    def test_a_sign_after_the_currency_symbol_is_a_negative_not_a_gap(self):
        frame = pd.DataFrame({"Amount": ["$-100"] + ["$200"] * 19})

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(float(cleaned["Amount"].sum()), 3700.0)
        self.assertEqual(int(cleaned["Amount"].isna().sum()), 0)
        self.assertEqual(report.numeric_cells_unreadable, 0)

    def test_every_correct_reading_is_kept(self):
        cases = {
            "-100": -100.0,
            "+100": 100.0,
            "1,000": 1000.0,
            "-$1,000.00": -1000.0,
            "(48.10)": -48.10,
            "1.234,50": 1234.5,
            "1 234,50": 1234.5,
            "1'234.50": 1234.5,
            "1.234.567": 1234567.0,
            "1,234": 1234.0,
            "12.5": 12.5,
            "1234,50": 1234.5,
            "€1.234,50": 1234.5,
            "£12,345": 12345.0,
            "$-100": -100.0,
            "-(100)": -100.0,
            "$(1,000.00)": -1000.0,
        }
        for text, value in cases.items():
            with self.subTest(text=text):
                self.assertEqual(_read_formatted_number(text), value)

    def test_malformed_numbers_are_refused_rather_than_read_past(self):
        malformed = (
            "(100", "100)", "1,2,3", "1.2.3", "12 34", "1,,000", ",100", "100,", "--100", "1,000.000,50"
        )
        for text in malformed:
            with self.subTest(text=text):
                self.assertIsNone(_read_formatted_number(text))

    def test_scientific_notation_is_plain_parsing_s_job(self):
        self.assertIsNone(_read_formatted_number("1e3"))
        cleaned, _ = clean_dataframe(pd.DataFrame({"Amount": ["1e3"] + ["1,000"] * 19}))
        self.assertEqual(cleaned["Amount"].tolist(), [1000.0] * 20)

    def test_a_cell_coerced_to_missing_is_counted_and_named(self):
        frame = pd.DataFrame({"Amount": ["(100"] + ["200"] * 19})

        cleaned, report = clean_dataframe(frame)

        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned["Amount"]))
        self.assertEqual(int(cleaned["Amount"].isna().sum()), 1)
        self.assertEqual(report.numeric_cells_unreadable, 1)
        self.assertIn("1 Amount values could not be read as numbers and were left missing.", report.notes)
        self.assertEqual(report.to_dict()["numeric_cells_unreadable"], 1)

    def test_blank_cells_do_not_count_against_the_numeric_bar(self):
        # A second column keeps the whitespace row in the frame; on its own it
        # would be dropped as an empty row before inference ever saw it.
        frame = pd.DataFrame({"Region": ["West", "East", "North"], "Amount": ["$100", "$200", "  "]})

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(cleaned["Amount"].tolist()[:2], [100.0, 200.0])
        self.assertTrue(pd.isna(cleaned["Amount"].iloc[2]))
        self.assertEqual(report.numeric_columns_inferred, 1)
        self.assertEqual(report.numeric_cells_unreadable, 0)

    def test_whole_numbers_past_exact_float_range_are_flagged_when_forced_to_float(self):
        frame = pd.DataFrame({"Balance": ["9007199254740993"] * 19 + ["$1"]})

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(cleaned["Balance"].iloc[0], 9007199254740992.0)
        flagged = [note for note in report.notes if "would not fit" in note and "Balance" in note]
        self.assertEqual(len(flagged), 1)

        exact, report = clean_dataframe(pd.DataFrame({"Balance": ["9007199254740993"] * 20}))

        self.assertEqual(int(exact["Balance"].iloc[0]), 9007199254740993)
        self.assertFalse(any("would not fit" in note for note in report.notes))

    def test_infinity_in_text_becomes_missing_and_is_counted(self):
        frame = pd.DataFrame({"Amount": ["inf"] + ["1"] * 19})

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(int(cleaned["Amount"].isna().sum()), 1)
        self.assertEqual(float(cleaned["Amount"].sum()), 19.0)
        self.assertEqual(report.non_finite_cells, 1)
        self.assertIn("1 Amount values were infinite and were left missing.", report.notes)

    def test_infinity_the_reader_already_parsed_is_caught_too(self):
        frame = read_tabular_file(b"Revenue\ninf\n-inf\n1\n", "f.csv")
        self.assertTrue(np.isinf(frame["Revenue"]).any())

        cleaned, report = clean_dataframe(frame)

        self.assertEqual(cleaned["Revenue"].tolist()[2], 1.0)
        self.assertEqual(int(cleaned["Revenue"].isna().sum()), 2)
        self.assertEqual(report.non_finite_cells, 2)
        self.assertIn("2 Revenue values were infinite and were left missing.", report.notes)


class TimezoneOffsetTests(unittest.TestCase):
    def test_a_rows_wall_clock_does_not_depend_on_the_rows_after_it(self):
        first, second = "2024-01-01T00:30:00+01:00", "2024-01-15T12:00:00+01:00"
        third = "2024-06-01T00:30:00+02:00"

        two, report_two = clean_dataframe(pd.DataFrame({"created_at": [first, second]}))
        three, report_three = clean_dataframe(pd.DataFrame({"created_at": [first, second, third]}))

        self.assertEqual(two["created_at"].tolist(), three["created_at"].tolist()[:2])
        self.assertEqual(two["created_at"].iloc[0], pd.Timestamp("2024-01-01 00:30:00"))
        self.assertEqual(three["created_at"].iloc[2], pd.Timestamp("2024-06-01 00:30:00"))
        self.assertFalse(isinstance(three["created_at"].dtype, pd.DatetimeTZDtype))
        for report in (report_two, report_three):
            self.assertEqual(report.notes.count(OFFSETS_DROPPED_NOTE), 1)

    def test_every_offset_spelling_is_dropped_the_same_way(self):
        frame = pd.DataFrame(
            {"When": ["2024-03-31T23:30:00+01:00", "2024-03-31T23:30:00+0100", "2024-03-31T23:30:00Z"]}
        )

        cleaned, _ = clean_dataframe(frame)

        self.assertEqual(cleaned["When"].tolist(), [pd.Timestamp("2024-03-31 23:30:00")] * 3)

    def test_a_column_without_offsets_gets_no_offset_note(self):
        frame = pd.DataFrame({"Order Date": ["2024-01-01", "2024-01-02"], "Revenue": [1, 2]})

        _, report = clean_dataframe(frame)

        self.assertNotIn(OFFSETS_DROPPED_NOTE, report.notes)


class MalformedFileTests(unittest.TestCase):
    def test_a_data_row_wider_than_the_header_is_refused_by_name(self):
        raw = b"Region,Revenue\nWest,100,999\nEast,200,888\n"

        with self.assertRaises(ValueError) as caught:
            read_tabular_file(raw, "sales.csv")

        message = str(caught.exception)
        self.assertIn("Line 2", message)
        self.assertIn("3 values", message)
        self.assertIn("2 columns", message)

    def test_a_trailing_delimiter_on_every_row_is_the_same_refusal(self):
        with self.assertRaises(ValueError):
            read_tabular_file(b"Region,Revenue\nWest,100,\nEast,200,\n", "sales.csv")

    def test_a_report_title_above_the_header_is_still_rescued(self):
        raw = b"Q3 Sales Report\nRegion,Revenue\nWest,100\nEast,200\n"

        frame = read_tabular_file(raw, "report.csv")

        self.assertEqual(list(frame.columns), ["Region", "Revenue"])
        self.assertEqual(frame["Region"].tolist(), ["West", "East"])

    def test_a_workbook_with_truncated_xml_is_a_readable_error(self):
        good = workbook_bytes([["Region", "Revenue"]] + [[f"R{i}", i] for i in range(50)])
        for member, cut, reader in (
            ("xl/workbook.xml", 40, list_excel_sheets),
            ("xl/workbook.xml", 40, read_tabular_file),
            ("xl/worksheets/sheet1.xml", 400, read_tabular_file),
        ):
            with self.subTest(member=member, reader=reader.__name__):
                with self.assertRaises(ValueError) as caught:
                    reader(truncated_member(good, member, cut), "book.xlsx")
                self.assertIn("damaged", str(caught.exception))

    def test_formulas_without_saved_results_are_refused_with_the_reason(self):
        contents = workbook_bytes(
            [["Region", "Units", "Revenue"], ["West", 2, "=B2*10"], ["East", 3, "=B3*10"]]
        )

        with self.assertRaises(ValueError) as caught:
            read_tabular_file(contents, "book.xlsx")

        message = str(caught.exception)
        self.assertIn("formulas without saved results", message)
        self.assertIn("Revenue", message)

    def test_a_genuinely_empty_column_is_not_mistaken_for_a_formula(self):
        contents = workbook_bytes([["Region", "Notes", "Revenue"], ["West", None, 10], ["East", None, 20]])

        frame = read_tabular_file(contents, "book.xlsx")

        self.assertEqual(frame["Revenue"].tolist(), [10, 20])
        self.assertTrue(frame["Notes"].isna().all())


class DateTiebreakTests(unittest.TestCase):
    def test_a_tie_between_long_names_resolves_the_same_in_either_column_order(self):
        stamps = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])
        columns = {
            "Transaction Date A": stamps,
            "Transaction Date B": stamps,
            "Revenue": [1, 2, 3],
        }

        forwards = detect_roles(pd.DataFrame(columns)).date
        backwards = detect_roles(pd.DataFrame(dict(reversed(columns.items())))).date

        self.assertEqual(forwards, "Transaction Date A")
        self.assertEqual(backwards, "Transaction Date A")


if __name__ == "__main__":
    unittest.main()
