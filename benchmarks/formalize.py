"""The ``solve`` condition of the benchmarks: the model writes the problem as equations, without solving it,
and ADA solves them exactly.

A small model often sets a problem up right and then slips in the arithmetic, or slips while carrying a
number from one step to the next. Here the model only translates the words into equations, and every step
of working them out is ADA's. The model names the quantity the question asks for on a last line,
"Answer: total", and ADA's value for it is the answer.

Models name quantities with words (``eggs_sold``) and ADA reads one letter as one unknown, so each name is
given a letter of its own before ADA reads the equations. ``e`` is ADA's constant, so a quantity named e is
given another letter too.
"""

from __future__ import annotations

import re
import string
from tokenize import TokenError

from ada.algebra import solve
from ada.compute import CONSTANTS, FUNCTION_NAMES, Refusal, compute
from ada.latex import LATEX, to_plain

FORMAL_PROMPT = (
    "Do not solve the problem. Write it as equations that an exact solver will solve. Put one equation on "
    "each line, and give each unknown quantity a name, such as x or eggs_sold. Write * for times, / for "
    "divide, ^ for powers and sqrt( ) for square roots. On the last line, write \"Answer:\" and the name or "
    "expression the question asks for.\n\n"
    "For example, for \"A shop sells 3 boxes of 12 pens and 5 more pens. How many pens does it sell?\":\n"
    "boxes = 3 * 12\n"
    "total = boxes + 5\n"
    "Answer: total"
)
_ANSWER = re.compile(r"^\W*answer\W*?:\s*(.*)$", re.IGNORECASE)
_NAME = re.compile(r"(?<![\\A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*")
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_DELIMITERS = re.compile(r"\$\$?|\\[()\[\]]|`+|\*\*")
_BOXED = re.compile(r"\\boxed\s*\{(.*)\}")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
#: A word or two in brackets after a value: "x = 5 (eggs)".
_ASIDE = re.compile(r"\s*\([A-Za-z\s]+\)\s*$")
#: Letters a quantity may be given: not e (ADA's constant) or i.
_LETTERS = [letter for letter in string.ascii_lowercase + string.ascii_uppercase if letter not in "eiE"]


def written(text: str) -> tuple[list[str], str | None]:
    """The equations a reply writes, one a line, and the quantity its "Answer:" line asks for."""
    equations, target = [], None
    for raw in text.splitlines():
        line = _BULLET.sub("", _DELIMITERS.sub("", raw)).strip()
        found = _ANSWER.match(line)
        if found:
            target = _target(found.group(1))
            continue
        if line.count("=") == 1 and not re.search(r"[<>!≈≠]", line):
            equations.append(_ASIDE.sub("", line.rstrip(".,;")).rstrip(".,;"))
    return equations, target


def _target(text: str) -> str | None:
    text = text.strip().rstrip(".")
    boxed = _BOXED.search(text)
    if boxed:
        text = boxed.group(1)
    if "=" in text:
        left, right = (side.strip() for side in text.split("=", 1))
        text = left if _NAME.fullmatch(left) else right
    return text or None


def lettered(equations: list[str], target: str | None) -> tuple[list[str], str | None, dict[str, str]]:
    """The equations and target with each name a model gave (``eggs_sold``, ``e``) changed to a letter of its
    own; function names and pi stay as they are."""
    names = []
    for text in equations + ([target] if target else []):
        for found in _NAME.finditer(text):
            name = found.group()
            reserved = name.lower() in FUNCTION_NAMES or name in CONSTANTS and name != "e"
            if not reserved and name not in names and (len(name) > 1 or name in "eiE"):
                names.append(name)
    taken = {found.group() for text in equations + ([target] if target else [])
             for found in _NAME.finditer(text) if len(found.group()) == 1}
    free = (letter for letter in _LETTERS if letter not in taken)
    mapping = {}
    for name in names:
        letter = next(free, None)
        if letter is None:
            break
        mapping[name] = letter

    def renamed(text: str) -> str:
        return _NAME.sub(lambda found: mapping.get(found.group(), found.group()), text)

    return [renamed(text) for text in equations], renamed(target) if target else None, mapping


def _plain(text: str) -> str:
    """A line of LaTeX in plain text, when it has any."""
    if LATEX.search(text) or "\\" in text:
        plain = to_plain(text, letters=True)
        return plain if plain is not None else text
    return text


def solved_by_ada(text: str, positive: bool = False) -> dict:
    """ADA's value for what a reply's equations ask for: the status, the value when there is exactly one, and
    what ADA read. With ``positive``, for word problems whose quantities are counts and amounts, a target
    with several values takes the one that is above zero, when only one is."""
    equations, target = written(text)
    if not target:
        return {"status": "no answer line", "value": None, "equations": equations, "target": None}
    renamed, aim, mapping = lettered(equations, target)
    renamed = [_plain(equation) for equation in renamed]
    aim = _plain(aim)
    record = {"equations": equations, "target": target, "names": mapping, "read": renamed, "aim": aim}
    sets = _solutions(renamed)
    if isinstance(sets, Refusal):
        return {**record, "status": "refused", "value": None, "message": sets.message}
    values = []
    for known in sets:
        value = _value_of(aim, known)
        if value is not None and value not in values:
            values.append(value)
    if len(values) > 1 and positive:
        above = [value for value in values if _sympy(value) is not None and _sympy(value) > 0]
        values = above if len(above) == 1 else values
    if len(values) == 1:
        # A target that is a bare number is the model's own answer, not ADA's.
        status = "number given" if _NUMBER.fullmatch(aim.replace(" ", "")) else "solved"
        return {**record, "status": status, "value": values[0]}
    return {**record, "status": "several values" if values else "no value", "value": None, "values": values}


def _solutions(equations: list[str]) -> list[dict[str, str]] | Refusal:
    """Each set of exact values ADA solves the equations to; no equations give one empty set."""
    if not equations:
        # "Answer: (16 - 3 - 4) * 2": an expression with no unknowns, worked out by ADA.
        return [{}]
    solution = solve(equations)
    if isinstance(solution, Refusal):
        # "r = sqrt(2)": ADA's algebra keeps to fractions, so a chain of assignments is worked out in order.
        chained = _assigned(equations)
        return solution if chained is None else [chained]
    found = solution.to_dict()
    if found["solutions"]:
        return [{name: value for name, value in entry["values"].items() if entry["exact"][name]}
                for entry in found["solutions"]]
    if found["roots"]:
        return [{found["unknowns"][0]: root["value"]} for root in found["roots"] if root["exact"]]
    return [{name: value for name, value in found["values"].items() if name not in found["free"]}]


def _assigned(equations: list[str]) -> dict[str, str] | None:
    """The values of a chain of assignments, each "letter = expression" in the letters before it, worked out
    exactly with SymPy; None when the equations are not such a chain."""
    known = {}
    for equation in equations:
        left, right = (side.strip() for side in equation.split("=", 1))
        if not re.fullmatch(r"[A-Za-z]", left) or left in known:
            return None
        value = _sympy(_filled(right, known))
        if value is None:
            return None
        known[left] = str(value)
    return known


def _filled(expression: str, known: dict[str, str]) -> str:
    return _NAME.sub(lambda found: f"({known[found.group()]})" if found.group() in known else found.group(),
                     expression)


def _value_of(aim: str, known: dict[str, str]) -> str | None:
    """The exact value of the target once the solved values are put in: ADA's when it is a fraction, else
    SymPy's, such as sqrt(2)/2."""
    filled = _filled(aim, known)
    result = compute(filled)
    if not isinstance(result, Refusal) and result.exact:
        return result.value
    value = _sympy(filled)
    return None if value is None else str(value)


def _sympy(expression: str):
    """The exact value of an expression with no unknowns left, with SymPy, or None."""
    import sympy  # noqa: PLC0415 - only the solve condition needs it
    from sympy.parsing.sympy_parser import (  # noqa: PLC0415
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    try:
        value = parse_expr(expression.replace("^", "**"), local_dict={"e": sympy.E, "pi": sympy.pi},
                           transformations=standard_transformations + (implicit_multiplication_application,))
        value = sympy.nsimplify(value, rational=False)
    except (sympy.SympifyError, TypeError, ValueError, SyntaxError, ZeroDivisionError, AttributeError,
            TokenError):
        return None
    if value.free_symbols or not value.is_number or value.has(sympy.zoo, sympy.nan, sympy.oo):
        return None
    return sympy.simplify(value)


def plain_number(value: str) -> str | None:
    """ADA's value as a plain decimal number: a fraction or a root is written to 12 significant figures."""
    if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        return value
    number = _sympy(value)
    return None if number is None or not number.is_real else format(float(number), ".12g")


def latex_value(value: str) -> str:
    """ADA's value in LaTeX, for a grader that reads it: 3*sqrt(3) is 3 \\sqrt{3}."""
    import sympy  # noqa: PLC0415 - only the solve condition needs it

    number = _sympy(value)
    return value if number is None else sympy.latex(number)
