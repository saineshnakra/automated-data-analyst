"""``ada init``: the design it proposes, and the analyses it writes only when the build would run them."""

import tempfile
import unittest
from pathlib import Path

from research_helpers import assay

from ada.research import init, load


class DesignConfirmationTests(unittest.TestCase):
    def test_a_design_accepted_with_yes_is_marked_unconfirmed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            assay().to_csv(root / "assay.csv", index=False)
            path = init(root / "assay.csv", study_path=root / "study.yml", interactive=False)
            self.assertIn("confirmed: false", path.read_text())
            self.assertFalse(load(path).design_confirmed)

    def test_answering_the_design_question_confirms_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            assay().to_csv(root / "assay.csv", index=False)
            path = init(
                root / "assay.csv", study_path=root / "study.yml", ask=lambda _: "", say=lambda _: None
            )
            self.assertTrue(load(path).design_confirmed)


if __name__ == "__main__":
    unittest.main()
