from __future__ import annotations

import unittest
import zipfile
from io import BytesIO

import pandas as pd

from ada import file_io
from ada.file_io import list_excel_sheets, list_sample_datasets, read_tabular_file, safe_csv


class FileParsingTests(unittest.TestCase):
    def test_reads_comma_separated_csv(self) -> None:
        result = read_tabular_file(b"Region,Revenue\nWest,1200\nEast,900\n", "sales.csv")

        self.assertEqual(result.columns.tolist(), ["Region", "Revenue"])
        self.assertEqual(result["Revenue"].sum(), 2100)

    def test_reads_tab_separated_files(self) -> None:
        contents = b"Patient\tDays to recovery\tDose, mg\nP1\t12\t5\nP2\t9\t10\n"

        for filename in ("Data.tsv", "export.txt"):
            with self.subTest(filename=filename):
                result = read_tabular_file(contents, filename)

                self.assertEqual(result.columns.tolist(), ["Patient", "Days to recovery", "Dose, mg"])
                self.assertEqual(result["Days to recovery"].sum(), 21)

    def test_reads_a_comma_separated_txt_file_as_a_csv(self) -> None:
        result = read_tabular_file(b"Region,Revenue\nWest,1200\nEast,900\n", "export.txt")

        self.assertEqual(result.columns.tolist(), ["Region", "Revenue"])

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

    def test_a_zip_that_is_not_a_workbook_is_a_readable_error(self):
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("hello.txt", "hi")

        for reader in (list_excel_sheets, read_tabular_file):
            with self.subTest(reader=reader.__name__), self.assertRaises(ValueError):
                reader(buffer.getvalue(), "x.xlsx")

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


class CellValueTests(unittest.TestCase):
    def test_na_is_north_america_not_missing(self):
        frame = read_tabular_file(b"Region,Revenue\nNA,10\nEU,20\n", "f.csv")

        self.assertEqual(frame["Region"].tolist(), ["NA", "EU"])

    def test_an_id_column_keeps_its_leading_zeros(self):
        frame = read_tabular_file(b"Customer ID,Revenue\n00042,10\n00007,20\n", "f.csv")

        self.assertEqual(frame["Customer ID"].tolist(), ["00042", "00007"])
        self.assertEqual(frame["Revenue"].tolist(), [10, 20])

    def test_a_database_null_is_missing(self):
        frame = read_tabular_file(b"code,laps\n\\N,\\N\nHAM,52\nVER,\\N\n", "results.csv")

        self.assertEqual(frame["code"].isna().tolist(), [True, False, False])
        self.assertEqual(frame["laps"].dropna().tolist(), [52])


class DelimiterRescueTests(unittest.TestCase):
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

    def test_a_large_semicolon_file_is_split_from_its_opening_lines(self):
        body = "".join(f"{n};A{n % 4};{n * 0.5}\n" for n in range(40_000)).encode()
        raw = b"index;Sample;Signal (mV)\n" + body

        frame = read_tabular_file(raw, "sweep.csv")

        self.assertGreater(len(raw), file_io.SNIFF_BYTES)
        self.assertEqual(list(frame.columns), ["index", "Sample", "Signal (mV)"])
        self.assertEqual(len(frame), 40_000)
        self.assertEqual(frame["Signal (mV)"].iloc[-1], 19_999.5)

    def test_a_quoted_line_break_at_the_sample_edge_still_reads(self):
        # The last line break in the opening bytes is inside a quoted note, so
        # the opening lines alone end mid-field and the whole file is used.
        opening = b'Id,Note\n' + b"1,short\n" * ((file_io.SNIFF_BYTES - 40) // 8)
        raw = opening + b'2,"a note\n' + b"x" * 60 + b' that spans lines"\n3,short\n'

        frame = read_tabular_file(raw, "notes.csv")

        self.assertGreater(len(raw), file_io.SNIFF_BYTES)
        self.assertEqual(list(frame.columns), ["Id", "Note"])
        self.assertTrue(frame["Note"].iloc[-2].startswith("a note\n"))

    def test_semicolons_split_a_file_whose_values_hold_commas(self):
        # Statistics Denmark: "Tickets sold (1,000)" holds a comma, so a comma read fails on a wider row.
        raw = ("\ufeffBIO;TID;INDHOLD\r\nTickets sold (1,000);2024;9500\r\nFilms (number);2024;489\r\n"
               "Seats (1,000, all);2024;12\r\n").encode("utf-8")

        frame = read_tabular_file(raw, "BIO5.csv")

        self.assertEqual(list(frame.columns), ["BIO", "TID", "INDHOLD"])
        self.assertEqual(frame["BIO"].tolist(),
                         ["Tickets sold (1,000)", "Films (number)", "Seats (1,000, all)"])

    def test_a_title_with_a_semicolon_above_comma_rows_is_skipped(self):
        raw = b"Source: Statistics; 2024\nDate,Region,Revenue\n2024-01-01,North,100\n2024-02-01,South,200\n"

        frame = read_tabular_file(raw, "report.csv")

        self.assertEqual(list(frame.columns), ["Date", "Region", "Revenue"])

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


class TitleRowEncodingTests(unittest.TestCase):
    def test_a_title_row_is_skipped_in_utf16_files(self):
        text = "Sales report Q1\nRegion,Revenue\nNorth,10\nSouth,20\n"
        for encoding in ("utf-8", "utf-16"):
            with self.subTest(encoding=encoding):
                frame = read_tabular_file(text.encode(encoding), "report.csv")
                self.assertEqual(list(frame.columns), ["Region", "Revenue"])


    def test_lines_a_data_portal_writes_above_the_header_are_skipped(self):
        text = (
            '"Data Source","World Development Indicators",\n\n"Last Updated Date","2026-10-08",\n\n'
            '"Country Name","Country Code","2020","2021",\n"Japan","JPN","84.6","84.4",\n'
            '"Chad","TCD","52.8","52.5",\n'
        )
        frame = read_tabular_file(text.encode(), "API_life.csv")
        self.assertEqual(list(frame.columns)[:4], ["Country Name", "Country Code", "2020", "2021"])
        self.assertEqual(frame["2021"].tolist(), [84.4, 52.5])

    def test_a_wider_row_further_down_is_not_taken_for_the_header(self):
        text = "a,b\n1,2\n3,4\n5,6,7\n"
        with self.assertRaises(ValueError):
            read_tabular_file(text.encode(), "ragged.csv")

class ArchiveSuffixTests(unittest.TestCase):
    def test_a_data_file_is_read_as_comma_separated_text(self) -> None:
        contents = b"subject#,age,sex,Jitter(%)\n1,72,0,0.00662\n1,72,0,0.003\n2,58,1,0.00481\n"

        result = read_tabular_file(contents, "parkinsons_updrs.data")

        self.assertEqual(result.columns.tolist(), ["subject#", "age", "sex", "Jitter(%)"])
        self.assertEqual(len(result), 3)

    def test_a_dat_file_is_read_as_tab_separated_when_its_header_splits(self) -> None:
        result = read_tabular_file(b"site\tdepth\nA\t1.5\nB\t2.0\n", "survey.dat")

        self.assertEqual(result["depth"].tolist(), [1.5, 2.0])

    def test_other_suffixes_are_still_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "ADA supports CSV, TSV, TXT, DATA, DAT, XLSX and XLSM files"):
            read_tabular_file(b"a,b\n1,2\n", "table.names")


if __name__ == "__main__":
    unittest.main()


class NotesSheetTests(unittest.TestCase):
    """A workbook that opens on a sheet of notes, as the Maddison Project's does."""

    @staticmethod
    def workbook(first: pd.DataFrame) -> bytes:
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            first.to_excel(writer, index=False, header=False, sheet_name="Notes")
            pd.DataFrame({"country": ["Chile", "Peru"], "year": [2021, 2022], "pop": [19.5, 33.7]}).to_excel(
                writer, index=False, sheet_name="Full data")
            pd.DataFrame({"Region": ["Andes"]}).to_excel(writer, index=False, sheet_name="Regions")
        return output.getvalue()

    def test_the_sheet_with_the_most_numbers_is_read_after_a_sheet_of_notes(self) -> None:
        policy = ("Citation policy: when the data are shown in any graphical form, cite the original papers "
                  "that were used to make the estimates, as listed on the sources sheet.")
        notes = pd.DataFrame({0: ["Maddison Project Database (MPD) 2023", policy]})
        frame = read_tabular_file(self.workbook(notes), "mpd.xlsx")
        self.assertEqual(frame.columns.tolist(), ["country", "year", "pop"])
        self.assertEqual(frame.attrs["sheet"], "Full data")

    def test_a_first_sheet_with_numbers_or_no_prose_is_read(self) -> None:
        figures = pd.DataFrame({0: ["Region", "West", "East"], 1: ["Revenue", 100, 200]})
        self.assertEqual(read_tabular_file(self.workbook(figures), "book.xlsx").columns.tolist(),
                         ["Region", "Revenue"])
        titles = pd.DataFrame({0: ["Contents", "Full data"]})
        self.assertNotIn("sheet", read_tabular_file(self.workbook(titles), "book.xlsx").attrs)

