"""The ``program`` condition of the GSM8K benchmark: the model writes a short Python program that works out
the answer, as in program-aided language models (PAL; Gao et al., 2023, the paper that made GSM-Hard), and the
program is run. It is the usual way of giving a model exact arithmetic without ADA, so ``solve`` is compared
with it.

Before a program is run, it is read and refused if it does more than arithmetic: it may use numbers, names,
arithmetic, comparisons, ``if``, ``for`` and ``while``, functions it defines, lists and the ``math`` module.
It may not import anything else, reach any attribute but ``math``'s, or use a name that starts with an
underscore, so it cannot read or write files or open connections. It runs in its own Python process, with a
few seconds of CPU and a limit on memory, and its answer is the number ``solution()`` returns. A float is
rounded to 6 decimal places, since a float carries rounding error (0.1 + 0.2 is 0.30000000000000004) and
PAL's own grading allows a difference of 0.001.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from fractions import Fraction

from formalize import FORMAL_EXAMPLES

PROGRAM_PROMPT = (
    "Write a Python program that solves the problem. Put the steps in a function named solution() that "
    "returns the number the question asks for, and give each quantity a name. Use only plain Python and the "
    "math module, and do not print anything. Write the program alone, in a ```python block."
)
#: The questions of FORMAL_EXAMPLES worked out as programs, so ``program`` has the same examples as ``solve``.
PROGRAMS = (
    "def solution():\n    price = 80\n    discount = price * 25 / 100\n    paid = price - discount - 5\n"
    "    return paid",
    "def solution():\n    together = 36\n    leo = together / 3\n    mia = 2 * leo\n    return mia",
    "def solution():\n    missing = 120 - 40\n    minutes = missing / 8\n    return minutes",
    "def solution():\n    x = (22 - 7) / 3\n    return 2 * x - 1",
)
#: The built-in functions a program may call.
BUILTINS = ("abs", "all", "any", "bool", "divmod", "enumerate", "float", "int", "len", "list", "map", "max",
            "min", "pow", "range", "reversed", "round", "sorted", "sum", "tuple", "zip")
#: The kinds of statement and expression a program may contain.
ALLOWED = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.Expr, ast.Name, ast.Load, ast.Store, ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.IfExp, ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass, ast.Call, ast.keyword,
    ast.Attribute, ast.List, ast.Tuple, ast.Subscript, ast.Slice, ast.ListComp, ast.GeneratorExp,
    ast.comprehension, ast.Import, ast.ImportFrom, ast.alias, ast.operator, ast.unaryop, ast.boolop,
    ast.cmpop,
)
#: Seconds of CPU a program may use, and bytes of memory.
CPU_SECONDS = 5
MEMORY = 512 * 1024 * 1024
DECIMALS = 6
_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

_RUNNER = """
import builtins, json, math, sys
def only_math(name, *args, **kwargs):
    if name != "math":
        raise ImportError("only the math module can be imported")
    return math
space = {"__builtins__": {**{name: getattr(builtins, name) for name in json.loads(sys.argv[1])},
                          "__import__": only_math}}
try:
    exec(compile(sys.stdin.read(), "<program>", "exec"), space)
    value = space["solution"]()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        print(json.dumps({"error": "solution() returned " + type(value).__name__ + ", not a number"}))
    elif isinstance(value, float) and not math.isfinite(value):
        print(json.dumps({"error": "solution() returned " + repr(value)}))
    else:
        print(json.dumps({"value": repr(value) if isinstance(value, float) else str(value)}))
except BaseException as error:
    print(json.dumps({"error": type(error).__name__ + ": " + str(error)[:300]}))
"""


def program_messages(question: str) -> list[dict]:
    """The chat for ``program``: the instruction, the examples written as programs, then the question."""
    messages = [{"role": "system", "content": PROGRAM_PROMPT}]
    for (asked, _), program in zip(FORMAL_EXAMPLES, PROGRAMS, strict=True):
        messages += [{"role": "user", "content": asked},
                     {"role": "assistant", "content": f"```python\n{program}\n```"}]
    return messages + [{"role": "user", "content": question}]


def program_in(reply: str) -> str | None:
    """The program in a reply: its last ```python block, or the whole reply when it defines solution()."""
    blocks = _BLOCK.findall(reply or "")
    if blocks:
        return blocks[-1].strip()
    return reply.strip() if "def solution" in (reply or "") else None


def refusal(program: str) -> str | None:
    """Why a program is not run, or None when it does only arithmetic."""
    try:
        tree = ast.parse(program)
    except (SyntaxError, ValueError) as error:
        return f"it is not Python: {error}"
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED):
            return f"it uses {type(node).__name__}, which a program here may not"
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            return f"it uses the name {node.id}"
        if isinstance(node, ast.Attribute) and not _math_attribute(node):
            return "it reaches an attribute other than the math module's"
        if isinstance(node, ast.Import) and any(alias.name != "math" for alias in node.names):
            return "it imports a module other than math"
        if isinstance(node, ast.ImportFrom) and node.module != "math":
            return "it imports a module other than math"
        if isinstance(node, ast.alias) and node.name.startswith("_"):
            return f"it imports {node.name}"
    return None


def _math_attribute(node: ast.Attribute) -> bool:
    """Whether an attribute is one of the math module's, such as math.sqrt."""
    return isinstance(node.value, ast.Name) and node.value.id == "math" and not node.attr.startswith("_")


def _limits() -> None:  # pragma: no cover - runs in the child process
    import resource  # noqa: PLC0415 - POSIX only

    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))


def run_program(reply: str) -> dict:
    """The answer of the program in a reply: ``status`` is "ran", "no program", "refused" or "failed", with
    ``value`` the number solution() returns as text, or None, and ``message`` saying why there is none."""
    program = program_in(reply)
    if program is None:
        return {"status": "no program", "value": None, "message": "the reply has no program"}
    reason = refusal(program)
    if reason:
        return {"status": "refused", "value": None, "message": f"The program is not run: {reason}."}
    try:
        finished = subprocess.run([sys.executable, "-I", "-S", "-c", _RUNNER, json.dumps(BUILTINS)],
                                  input=program, text=True,
                                  capture_output=True, timeout=CPU_SECONDS * 3,
                                  preexec_fn=_limits if sys.platform != "win32" else None)
        found = json.loads(finished.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"status": "failed", "value": None, "message": "the program ran too long"}
    except (IndexError, ValueError):
        stopped = finished.returncode < 0 if sys.platform != "win32" else False
        message = (f"the program was stopped: it used more than {CPU_SECONDS} seconds of CPU or "
                   f"{MEMORY // 2**20} MB of memory" if stopped else "the program stopped without an answer")
        return {"status": "failed", "value": None, "message": message}
    if "error" in found:
        return {"status": "failed", "value": None, "message": found["error"]}
    return {"status": "ran", "value": rounded(found["value"]), "message": ""}


def rounded(value: str) -> str:
    """A program's number as a plain decimal: a whole number as it is, a float to DECIMALS places."""
    number = round(Fraction(value), DECIMALS)
    if number.denominator == 1:
        return str(number.numerator)
    sign, number = ("-", -number) if number < 0 else ("", number)
    whole, part = divmod(number.numerator * 10**DECIMALS // number.denominator, 10**DECIMALS)
    return f"{sign}{whole}.{part:0{DECIMALS}d}".rstrip("0").rstrip(".")
