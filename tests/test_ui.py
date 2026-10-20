import unittest

import pandas as pd

from ada.ui import group_names


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
