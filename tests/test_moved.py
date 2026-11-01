"""The research workflow, the modelling mode and the app moved to ADA Lab: ADA hands them on or says how."""

import contextlib
import io
import sys
import types
import unittest
from unittest import mock

import ada
from ada.cli import MOVED_COMMANDS, main


def _without_ada_lab():
    """Make ``import ada_lab`` fail, as it does where ADA Lab is not installed."""
    return mock.patch.dict(sys.modules, {"ada_lab": None, "ada_lab.cli": None, "ada_lab.research": None})


class MovedCommandTests(unittest.TestCase):
    def test_a_moved_command_says_how_to_get_ada_lab(self):
        for command in MOVED_COMMANDS:
            err = io.StringIO()
            with _without_ada_lab(), contextlib.redirect_stderr(err):
                code = main([command, "--help"])
            self.assertEqual(code, 2)
            self.assertIn("pip install ada-lab", err.getvalue())
            self.assertIn(f"ada-lab {command}", err.getvalue())

    def test_a_moved_command_runs_in_ada_lab_when_it_is_installed(self):
        calls = []
        lab = types.ModuleType("ada_lab")
        cli = types.ModuleType("ada_lab.cli")
        cli.main = lambda argv, prog: calls.append((argv, prog)) or 0
        with mock.patch.dict(sys.modules, {"ada_lab": lab, "ada_lab.cli": cli}):
            self.assertEqual(main(["build", "--strict"]), 0)
        self.assertEqual(calls, [(["build", "--strict"], "ada")])

    def test_no_command_prints_the_help(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main([]), 0)
        text = " ".join(out.getvalue().split())
        self.assertIn("usage: ada", text)
        self.assertIn("are in ADA Lab: pip install ada-lab.", text)

    def test_the_core_commands_are_not_handed_on(self):
        out = io.StringIO()
        with _without_ada_lab(), contextlib.redirect_stdout(out):
            self.assertEqual(main(["samples"]), 0)
        self.assertIn("lab-measurements.csv", out.getvalue())


class MovedNameTests(unittest.TestCase):
    def test_a_moved_name_says_how_to_get_ada_lab(self):
        for name in ("model", "ModelReport", "research"):
            with _without_ada_lab(), self.assertRaises(ImportError) as raised:
                getattr(ada, name)
            self.assertIn("pip install ada-lab", str(raised.exception))
        with _without_ada_lab(), self.assertRaises(ImportError):
            from ada import research  # noqa: F401

    def test_a_moved_name_is_ada_labs_when_it_is_installed(self):
        lab = types.ModuleType("ada_lab")
        lab.model = object()
        research = types.ModuleType("ada_lab.research")
        with mock.patch.dict(sys.modules, {"ada_lab": lab, "ada_lab.research": research}):
            self.assertIs(ada.model, lab.model)
            self.assertIs(ada.research, research)

    def test_other_missing_names_are_attribute_errors(self):
        self.assertFalse(hasattr(ada, "no_such_name"))
        with self.assertRaises(AttributeError):
            ada.no_such_name  # noqa: B018


if __name__ == "__main__":
    unittest.main()
