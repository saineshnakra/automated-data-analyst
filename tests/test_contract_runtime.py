"""What a long-lived server process serves after the code under it changes.

Streamlit re-executes app.py on every rerun and reloads a stale module before
any local import, but a reload is only half of a deploy: the parsed uploads
are cached, the build label is cached, and the chat transcript is keyed on
the dataset alone. Each of these could keep serving the previous revision
after the reload had already replaced it. These tests reproduce that in a
subprocess -- one process, two runs, source changed between them -- so the
test process's own modules are never touched.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import pandas as pd

from schema import ColumnRoles

LABEL = re.compile(r"^[0-9a-f]{7}$")


def _run_in_scratch(script: str) -> subprocess.CompletedProcess:
    """Run a simulation script against a private copy of the checkout."""
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as scratch:
        for path in root.glob("*.py"):
            shutil.copy(path, scratch)
        for folder in ("samples", ".streamlit", "assets"):
            if (root / folder).exists():
                shutil.copytree(root / folder, Path(scratch) / folder)
        return subprocess.run(
            [sys.executable, "-c", script],
            cwd=scratch,
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "PYTHONPATH": scratch},
        )


def _report(result: subprocess.CompletedProcess) -> dict:
    if result.returncode != 0:
        raise AssertionError(result.stderr[-3000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


class UploadCacheTests(unittest.TestCase):
    """A reloaded reader must not be answered from the old reader's cache."""

    SIM = textwrap.dedent(
        r"""
        import sys, runpy, warnings, logging, io, contextlib, json
        warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
        def run():
            with contextlib.redirect_stderr(io.StringIO()):
                return runpy.run_path("app.py", run_name="__main__")
        contents = b"Revenue\n10\n20\n"
        page = run()
        label1 = page["build_identifier"]()
        first = page["read_uploaded_file"](contents, "t.csv", None, label1)["Revenue"].tolist()
        with open("file_io.py", "a") as handle:
            handle.write(
                "\n\ndef read_tabular_file(contents, filename, sheet_name=None):\n"
                "    import pandas as pd\n"
                "    return pd.DataFrame({'Revenue': [100, 200]})\n"
            )
        page = run()
        label2 = page["build_identifier"]()
        second = page["read_uploaded_file"](contents, "t.csv", None, label2)["Revenue"].tolist()
        print(json.dumps({"first": first, "second": second, "label1": label1, "label2": label2}))
        """
    )

    def test_a_changed_reader_is_a_cache_miss_and_a_new_build_label(self):
        report = _report(_run_in_scratch(self.SIM))

        self.assertEqual(report["first"], [10, 20])
        self.assertEqual(report["second"], [100, 200])
        self.assertNotEqual(report["label1"], report["label2"])
        # The scratch copy has no checkout, so the label is the digest alone.
        self.assertRegex(report["label1"], LABEL)
        self.assertRegex(report["label2"], LABEL)


class BuildLabelTests(unittest.TestCase):
    """The build label says what is served, and the commit is only a suffix."""

    def setUp(self):
        import app  # noqa: PLC0415 - importing runs the page once, in bare mode

        self.app = app

    def test_the_label_is_the_source_digest_with_the_commit_appended(self):
        digest = hashlib.sha256()
        for name in self.app._LOCAL_MODULES + ("app",):
            digest.update(self.app._source_digest(name).encode())
        sha = self.app._git_short_sha()
        expected = digest.hexdigest()[:7] + (f"@{sha}" if sha else "")

        self.assertEqual(self.app.build_identifier(), expected)

    def test_the_commit_is_read_from_a_checkout(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / ".git" / "refs" / "heads").mkdir(parents=True)
            (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
            head = root / ".git" / "refs" / "heads" / "main"
            head.write_text("47b45020248c4021a0e1c0413409ee4e7a4ab49e\n")
            self.assertEqual(self.app._git_short_sha(root), "47b4502")

            (root / ".git" / "HEAD").write_text("0d12cc65a514a4dee6d483f260cd604676adea7e\n")
            self.assertEqual(self.app._git_short_sha(root), "0d12cc6")

        with tempfile.TemporaryDirectory() as scratch:
            self.assertEqual(self.app._git_short_sha(Path(scratch)), "")


class RevisionIdentityTests(unittest.TestCase):
    """An answer, or a pending plan, must not survive the code that made it."""

    def setUp(self):
        import app  # noqa: PLC0415 - importing runs the page once, in bare mode

        self.app = app
        self.roles = ColumnRoles(
            date=None, measure="revenue", dimension="region",
            identifier=None, numeric=("revenue",), dimensions=("region",),
        )
        self.frame = pd.DataFrame({"region": ["N", "S"], "revenue": [1.0, 2.0]})

    def test_a_new_build_is_a_different_dataset(self):
        before = self.app.dataset_fingerprint(self.frame, self.roles, "q.csv")
        original = self.app.build_identifier
        self.app.build_identifier = lambda: "0000000@1111111"
        try:
            after = self.app.dataset_fingerprint(self.frame, self.roles, "q.csv")
            again = self.app.dataset_fingerprint(self.frame, self.roles, "q.csv")
        finally:
            self.app.build_identifier = original

        self.assertNotEqual(before, after)
        self.assertEqual(after, again)
        self.assertEqual(before, self.app.dataset_fingerprint(self.frame, self.roles, "q.csv"))


class WarmReloadFailureTests(unittest.TestCase):
    """A deploy that breaks a module must fail the way a cold start would.

    The optional AI layer is guarded at import, but a stale copy of it is
    reloaded before that guard runs. If the reload raises, the page is down
    for a dependency the product does not need. A broken deterministic module
    is a real outage, and the page must say which file, not show the redacted
    traceback the host serves.
    """

    SIM = textwrap.dedent(
        r"""
        import json, warnings, logging
        warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
        from streamlit.testing.v1 import AppTest
        first = AppTest.from_file("app.py", default_timeout=180).run()
        broken = 'raise ImportError("simulated: %(module)s is broken on this deploy")\n'
        open("%(module)s.py", "w").write(broken)
        second = AppTest.from_file("app.py", default_timeout=180).run()
        print(json.dumps({
            "first_exceptions": [str(e.value) for e in first.exception],
            "first_warnings": [str(w.value) for w in first.warning],
            "second_exceptions": [str(e.value) for e in second.exception],
            "second_warnings": [str(w.value) for w in second.warning],
            "second_captions": [str(c.value) for c in second.caption],
            "second_errors": [str(e.value) for e in second.error],
            "second_tabs": len(second.tabs),
        }))
        """
    )
    AI_WARNING = "optional AI layer could not be loaded"

    def test_a_broken_optional_module_degrades_the_page_instead_of_crashing_it(self):
        report = _report(_run_in_scratch(self.SIM % {"module": "ai_insights"}))

        self.assertEqual(report["first_exceptions"], [])
        self.assertFalse(any(self.AI_WARNING in text for text in report["first_warnings"]))

        self.assertEqual(report["second_exceptions"], [])
        self.assertTrue(any(self.AI_WARNING in text for text in report["second_warnings"]))
        self.assertIn(
            "ImportError: simulated: ai_insights is broken on this deploy",
            report["second_captions"],
        )
        # The deterministic product is untouched.
        self.assertEqual(report["second_tabs"], 6)

    def test_a_broken_deterministic_module_names_itself_and_still_fails(self):
        report = _report(_run_in_scratch(self.SIM % {"module": "formatting"}))

        self.assertEqual(report["first_exceptions"], [])
        self.assertEqual(len(report["second_exceptions"]), 1)
        self.assertIn("simulated: formatting is broken on this deploy", report["second_exceptions"][0])
        self.assertEqual(
            [text for text in report["second_errors"] if "formatting.py" in text],
            [
                "ADA could not load the deployed formatting.py into the running server: "
                "ImportError: simulated: formatting is broken on this deploy. The page will "
                "not render until the file is fixed or the server is restarted."
            ],
        )


if __name__ == "__main__":
    unittest.main()
