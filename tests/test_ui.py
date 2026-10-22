import unittest

import pandas as pd

from ada.autovis import recommend_chart
from ada.formatting import UNDATED_DAY
from ada.ui import _explore_figure, _explore_frame, group_names


class UndatedAxisTests(unittest.TestCase):
    def test_an_explore_chart_of_times_with_no_date_labels_the_clock(self):
        frame = pd.DataFrame({
            "Time": UNDATED_DAY + pd.to_timedelta(range(54_388, 54_388 + 60), unit="s"),
            "Signal": [float(n % 7) for n in range(60)],
        })
        spec = recommend_chart(frame, ["Time", "Signal"])

        figure = _explore_figure(_explore_frame(frame, spec), spec)

        self.assertEqual(figure.layout.xaxis.tickformat, "%H:%M:%S")


class GroupNameTests(unittest.TestCase):
    def test_numeric_codes_are_named_with_their_column(self):
        frame = pd.DataFrame({"Status": [0, 1, 2]})

        self.assertEqual(group_names(frame, "Status", [2, 0]), ["Status 2", "Status 0"])
        # The aggregations may return the codes as text.
        self.assertEqual(
            group_names(frame, "Status", ["2", "Other segments"]), ["Status 2", "Other segments"]
        )

    def test_text_groups_are_left_as_they_are(self):
        frame = pd.DataFrame({"Site": ["North", "South"]})

        self.assertEqual(group_names(frame, "Site", ["North", "South"]), ["North", "South"])


if __name__ == "__main__":
    unittest.main()
