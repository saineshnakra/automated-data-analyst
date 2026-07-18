from __future__ import annotations

import unittest
from io import BytesIO

import pandas as pd

from file_io import read_tabular_file


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

    def test_rejects_empty_and_unsupported_files(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            read_tabular_file(b"", "sales.csv")
        with self.assertRaisesRegex(ValueError, "supports"):
            read_tabular_file(b"content", "sales.json")


if __name__ == "__main__":
    unittest.main()
