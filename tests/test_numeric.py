import re
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import ada
from ada.numeric import to_numbers

# A hashed identifier from a public order table: pandas 2.3 reads "689e8820798347950" in it as an
# exponent past its parser's limit and crashes the process.
HASHED_ID = "689e8820798347950a2649f71c734bd0"


def run(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter, so a crash fails the test instead of the test run."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=300,
        cwd=Path(ada.__file__).parent.parent,
    )


class ToNumbersTests(unittest.TestCase):
    def test_it_matches_pandas_on_ordinary_text(self):
        values = pd.Series(["1.5", "x", None, "2e3", " 7 ", "-0.25"], index=[5, 6, 7, 8, 9, 10], dtype=object)
        pd.testing.assert_series_equal(to_numbers(values), pd.to_numeric(values, errors="coerce"))

    def test_an_oversized_exponent_is_missing_without_a_crash(self):
        result = run(f"""
            import pandas as pd
            from ada.numeric import to_numbers
            print(to_numbers(pd.Series([{HASHED_ID!r}, "1e2147483648", "1E+2147483648", "2.5"])).tolist())
        """)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(result.stdout.strip(), "[nan, nan, nan, 2.5]")

    def test_it_returns_what_pandas_returns(self):
        self.assertIsInstance(to_numbers(pd.Index(["1", "2"])), pd.Index)
        self.assertIsInstance(to_numbers(np.array(["1", HASHED_ID], dtype=object)), np.ndarray)
        self.assertTrue(np.isnan(to_numbers([HASHED_ID, "3"])[0]))
        numbers = pd.Series([1, 2, 3])
        pd.testing.assert_series_equal(to_numbers(numbers), numbers)

    def test_leading_zeros_in_an_exponent_are_not_oversized(self):
        self.assertEqual(to_numbers(pd.Series(["1e0000000000002"])).tolist(), [100.0])

    def test_no_module_calls_pandas_directly(self):
        root = Path(ada.__file__).parent
        calls = [
            f"{path.relative_to(root)}:{number}"
            for path in sorted(root.rglob("*.py")) if path.name != "numeric.py"
            for number, line in enumerate(path.read_text().splitlines(), 1)
            if re.search(r"\bto_numeric\(", line)
        ]
        self.assertEqual(calls, [], "use ada.numeric.to_numbers instead of pd.to_numeric")


class HashedIdentifierTests(unittest.TestCase):
    def test_a_table_with_hashed_identifiers_is_analyzed(self):
        result = run(f"""
            import pandas as pd
            import ada
            frame = pd.DataFrame({{
                "order_id": [{HASHED_ID!r}] + [f"{{n:032x}}" for n in range(1, 60)],
                "payment_value": [float(n) + 0.5 for n in range(60)],
            }})
            report = ada.analyze(frame)
            print(report.rows_analyzed, ada.check_claim(report, "The total payment value is 1800.").status)
        """)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(result.stdout.strip().splitlines()[-1], "60 verified")


if __name__ == "__main__":
    unittest.main()
