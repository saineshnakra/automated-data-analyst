import unittest

import pandas as pd

from ada.autovis import recommend_chart
from ada.formatting import UNDATED_DAY
from ada.nlq import answer_question
from ada.schema import detect_roles
from ada.ui import _chat_answer_figure, _explore_figure, _explore_frame, group_names


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


class ChatFigureTests(unittest.TestCase):
    def test_binned_groups_draw_and_a_table_of_statistics_does_not(self):
        frame = pd.DataFrame({
            "Growth": [float(index % 13) for index in range(80)],
            "Temperature": [index / 3 for index in range(80)],
        })
        roles = detect_roles(frame)

        binned = answer_question("average Growth by Temperature", frame, roles)
        described = answer_question("distribution of Growth", frame, roles)

        self.assertIsNotNone(_chat_answer_figure(binned))
        self.assertIsNone(_chat_answer_figure(described))


if __name__ == "__main__":
    unittest.main()
