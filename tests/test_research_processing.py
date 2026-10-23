import unittest

import pandas as pd
from research_helpers import assay, assay_study, study_folder

from ada.research.audit import audit
from ada.research.processing import SOURCE_ROW, read
from ada.research.study import StudyError, load


class ProcessingTests(unittest.TestCase):
    def test_codes_become_missing_and_are_counted(self):
        with study_folder(assay(), assay_study()) as path:
            processed = read(load(path))
        self.assertTrue(pd.isna(processed.data.loc[2, "Viability (%)"]))
        self.assertEqual(processed.missing_codes["Viability (%)"], {"-99": 1})
        self.assertEqual([record.kind for record in processed.records], ["read", "missing-codes"])

    def test_an_exclusion_keeps_the_removed_rows_and_their_lines(self):
        steps = [
            {"exclude": {"where": {"column": "Viability (%)", "below": 0}}, "reason": "Instrument error."}
        ]
        with study_folder(assay(), assay_study(processing=steps)) as path:
            processed = read(load(path))
        record = processed.records[-1]
        self.assertEqual((record.rows_before, record.rows_after), (36, 35))
        self.assertEqual(record.reason, "Instrument error.")
        self.assertEqual(processed.excluded[SOURCE_ROW].tolist(), [17])
        self.assertEqual(processed.excluded["Well"].tolist(), ["L03"])

    def test_ids_exclude_by_the_named_column(self):
        steps = [{"exclude": {"ids": ["C00", "C01"], "id_column": "Well"}, "reason": "Contaminated."}]
        with study_folder(assay(), assay_study(processing=steps)) as path:
            processed = read(load(path))
        self.assertEqual(processed.records[-1].removed_ids, ("C00", "C01"))
        self.assertEqual(len(processed.data), 34)

    def test_transform_and_recode_write_new_columns(self):
        study = assay_study(
            processing=[
                {
                    "transform": {"column": "Passage", "to": "log_passage", "apply": "log10"},
                    "reason": "Skewed.",
                },
                {
                    "recode": {
                        "column": "Treatment",
                        "map": {"Low": "Dose", "High": "Dose"},
                        "to": "Any dose",
                    },
                    "reason": "Pool the doses.",
                },
            ]
        )
        study["variables"]["log_passage"] = {"role": "covariate", "type": "numeric"}
        study["variables"]["Any dose"] = {"role": "group", "type": "categorical"}
        with study_folder(assay(), study) as path:
            processed = read(load(path))
        self.assertAlmostEqual(
            processed.data["log_passage"].iloc[0],
            pd.Series(processed.data["Passage"]).apply(lambda v: __import__("math").log10(v)).iloc[0],
        )
        self.assertEqual(set(processed.data["Any dose"].astype(str)), {"Control", "Dose"})
        self.assertEqual(processed.records[-1].values_changed, 24)

    def test_levels_follow_the_study_order(self):
        with study_folder(assay().iloc[::-1], assay_study()) as path:
            processed = read(load(path))
        self.assertEqual(list(processed.data["Treatment"].cat.categories), ["Control", "Low", "High"])

    def test_a_changed_file_stops_the_build(self):
        study = assay_study()
        study["data"]["sha256"] = "0" * 64
        with study_folder(assay(), study) as path, self.assertRaises(StudyError) as raised:
            read(load(path))
        self.assertIn("has changed since study.yml recorded it", str(raised.exception))

    def test_a_described_column_must_exist(self):
        study = assay_study()
        study["variables"]["Dose"] = {"role": "covariate", "type": "numeric"}
        with study_folder(assay(), study) as path, self.assertRaises(StudyError) as raised:
            read(load(path))
        self.assertIn("has no such column", str(raised.exception))


class AuditTests(unittest.TestCase):
    def test_values_outside_the_range_are_listed_with_their_lines(self):
        with study_folder(assay(), assay_study()) as path:
            study = load(path)
            findings = audit(study, read(study))
        outside = [finding for finding in findings if "outside the allowed range" in finding.finding]
        self.assertEqual(len(outside), 1)
        self.assertEqual(outside[0].where, "lines 17")
        self.assertEqual(outside[0].status, "check")

    def test_an_independent_design_with_repeated_subjects_is_questioned(self):
        frame = assay()
        frame["Well"] = frame["Well"].str[:2]
        study = assay_study(design={"type": "independent", "subject": "Well"})
        with study_folder(frame, study) as path:
            findings = audit(load(path), read(load(path)))
        self.assertTrue(any("set `design.type: repeated`" in finding.finding for finding in findings))

    def test_a_paired_design_lists_subjects_missing_a_condition(self):
        frame = pd.DataFrame(
            {
                "Subject": ["a", "a", "b", "b", "c"],
                "Session": ["pre", "post"] * 2 + ["pre"],
                "Score": [1.0, 2.0, 1.5, 2.5, 1.2],
            }
        )
        study = {
            "study": {"title": "t"},
            "data": {"path": "assay.csv", "observation": "one session"},
            "design": {"type": "paired", "subject": "Subject"},
            "variables": {
                "Subject": {"role": "identifier", "type": "text"},
                "Session": {"role": "group", "type": "categorical", "levels": ["pre", "post"]},
                "Score": {"role": "outcome", "type": "numeric"},
            },
        }
        with study_folder(frame, study) as path:
            findings = audit(load(path), read(load(path)))
        paired = next(finding for finding in findings if "have both conditions" in finding.finding)
        self.assertIn("1 have only one and cannot be paired: c", paired.finding)

    def test_hourly_readings_at_each_station_are_not_repeats_and_their_hours_are_summarised(self):
        stamps = pd.date_range("2016-12-29", periods=5 * 24, freq="h")
        frame = pd.concat(
            pd.DataFrame({
                "year": stamps.year, "month": stamps.month, "day": stamps.day, "hour": stamps.hour,
                "PM2.5": [float(10 + stamp.hour % 6) for stamp in stamps], "station": station,
            })
            for station in ("Aotizhongxin", "Changping")
        )
        study = {
            "study": {"title": "t"},
            "data": {"path": "air.csv", "observation": "one station at one hour"},
            "design": {"type": "repeated", "subject": "station", "time": "year-month-day-hour"},
            "variables": {
                "year-month-day-hour": {"role": "time", "type": "date"},
                "station": {"role": "identifier", "type": "text"},
                "PM2.5": {"role": "outcome", "type": "numeric"},
            },
        }
        with study_folder(frame, study, "air.csv") as path:
            findings = [finding.finding for finding in audit(load(path), read(load(path)))]
        self.assertFalse(any("repeat a subject" in finding for finding in findings))
        self.assertIn(
            "120 time points from 2016-12-29 00:00:00 to 2017-01-02 23:00:00, "
            "with 2 subjects measured at each.",
            findings,
        )


if __name__ == "__main__":
    unittest.main()
