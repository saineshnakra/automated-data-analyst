from __future__ import annotations

import unittest
from io import BytesIO

import pandas as pd

from file_io import list_excel_sheets, read_tabular_file


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


if __name__ == "__main__":
    unittest.main()
