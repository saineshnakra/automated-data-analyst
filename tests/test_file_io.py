from __future__ import annotations

import unittest
from io import BytesIO

import pandas as pd

from file_io import list_excel_sheets, list_sample_datasets, read_tabular_file, safe_csv


class FileParsingTests(unittest.TestCase):
    def test_reads_comma_separated_csv(self) -> None:
        result = read_tabular_file(b"Region,Revenue\nWest,1200\nEast,900\n", "sales.csv")

        self.assertEqual(result.columns.tolist(), ["Region", "Revenue"])
        self.assertEqual(result["Revenue"].sum(), 2100)

    def test_detects_semicolon_delimiter(self) -> None:
        result = read_tabular_file(b"Region;Revenue\nWest;1200\nEast;900\n", "sales.csv")

        self.assertEqual(result.columns.tolist(), ["Region", "Revenue"])
        self.assertEqual(len(result), 2)

    def test_reads_first_excel_worksheet(self) -> None:
        output = BytesIO()
        source = pd.DataFrame({"Product": ["Core", "Plus"], "Revenue": [800, 1200]})
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            source.to_excel(writer, index=False, sheet_name="Operating data")

        result = read_tabular_file(output.getvalue(), "business.xlsx")

        pd.testing.assert_frame_equal(result, source)

    def _multi_sheet_workbook(self) -> bytes:
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pd.DataFrame({"Region": ["West"], "Revenue": [100]}).to_excel(
                writer, index=False, sheet_name="Sales"
            )
            pd.DataFrame({"Team": ["Support"], "Tickets": [42]}).to_excel(
                writer, index=False, sheet_name="Operations"
            )
        return output.getvalue()

    def test_lists_worksheets_of_a_workbook(self) -> None:
        workbook = self._multi_sheet_workbook()

        self.assertEqual(list_excel_sheets(workbook, "book.xlsx"), ["Sales", "Operations"])
        self.assertEqual(list_excel_sheets(b"Region,Revenue\nWest,1\n", "sales.csv"), [])

    def test_reads_a_chosen_worksheet(self) -> None:
        workbook = self._multi_sheet_workbook()

        chosen = read_tabular_file(workbook, "book.xlsx", sheet_name="Operations")
        default = read_tabular_file(workbook, "book.xlsx")

        self.assertEqual(chosen.columns.tolist(), ["Team", "Tickets"])
        self.assertEqual(default.columns.tolist(), ["Region", "Revenue"])

    def test_corrupt_workbook_raises_a_friendly_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a valid Excel workbook"):
            list_excel_sheets(b"definitely not a zip", "book.xlsx")
        with self.assertRaisesRegex(ValueError, "not a valid Excel workbook"):
            read_tabular_file(b"definitely not a zip", "book.xlsx")

    def test_rejects_empty_and_unsupported_files(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            read_tabular_file(b"", "sales.csv")
        with self.assertRaisesRegex(ValueError, "supports"):
            read_tabular_file(b"content", "sales.json")

    def test_bundled_samples_are_listed_and_readable(self):
        samples = list_sample_datasets()

        self.assertGreaterEqual(len(samples), 3)
        self.assertIn("SaaS Subscriptions", samples)
        for name, path in samples.items():
            with self.subTest(sample=name):
                frame = read_tabular_file(path.read_bytes(), path.name)
                self.assertGreater(len(frame), 0)
                self.assertGreater(len(frame.columns), 1)



class DelimiterRescueTests(unittest.TestCase):
    """The rescue for European CSVs must not shred ordinary one-column files."""

    def test_a_semicolon_delimited_file_is_still_split(self):
        frame = read_tabular_file(b"Datum;Region;Umsatz\n2024-01-01;Nord;100\n", "eu.csv")

        self.assertEqual(list(frame.columns), ["Datum", "Region", "Umsatz"])

    def test_a_separator_in_the_body_alone_does_not_split_the_file(self):
        raw = b"Feedback\nGreat; really good\nSlow; but fine\n"

        frame = read_tabular_file(raw, "notes.csv")

        self.assertEqual(list(frame.columns), ["Feedback"])
        self.assertEqual(len(frame), 2)

    def test_a_tab_in_one_cell_does_not_split_the_file(self):
        frame = read_tabular_file(b"Revenue\n100\n200\n500\tX\n", "one.csv")

        self.assertEqual(list(frame.columns), ["Revenue"])

    def test_a_report_title_above_the_header_is_skipped(self):
        raw = b"Q3 Sales Report\nDate,Region,Revenue\n2024-01-01,North,100\n2024-02-01,South,200\n"

        frame = read_tabular_file(raw, "report.csv")

        self.assertEqual(list(frame.columns), ["Date", "Region", "Revenue"])
        self.assertEqual(len(frame), 2)

    def test_a_utf16_export_is_decoded_by_its_byte_order_mark(self):
        raw = "Date,Region,Revenue\n2024-01-01,North,100\n".encode("utf-16")

        frame = read_tabular_file(raw, "powershell.csv")

        self.assertEqual(list(frame.columns), ["Date", "Region", "Revenue"])



class CsvExportSafetyTests(unittest.TestCase):
    """A downloaded file gets forwarded, and the reader did not choose to run anything."""

    def test_text_that_looks_like_a_formula_is_neutralised(self):
        frame = pd.DataFrame({"Note": ["=cmd|' /C calc'!A0", "+1+1", "@SUM(A1)", "-5 apples"]})

        exported = safe_csv(frame)

        for line in exported.splitlines()[1:]:
            self.assertTrue(line.startswith("'"), line)

    def test_ordinary_text_and_numbers_are_untouched(self):
        frame = pd.DataFrame({"Region": ["North", "South"], "Revenue": [-2.5, 3.0]})

        exported = safe_csv(frame)

        self.assertIn("North", exported)
        self.assertIn("-2.5", exported)
        self.assertNotIn("'North", exported)
        self.assertNotIn("'-2.5", exported)

    def test_a_column_name_that_looks_like_a_formula_is_neutralised(self):
        frame = pd.DataFrame({"=1+1": [1]})

        self.assertTrue(safe_csv(frame).startswith("'=1+1"))

if __name__ == "__main__":
    unittest.main()