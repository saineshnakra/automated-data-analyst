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

import json
import math
import re
import string
from decimal import Decimal, localcontext

from ada.algebra import solve
from ada.compute import CONSTANTS, FUNCTION_NAMES, Refusal, compute
from ada.latex import LATEX, to_plain

FORMAL_PROMPT = (
    "Do not solve the problem. Write it as equations that an exact solver will solve. Put one equation on "
    "each line, and give each quantity a name of one word, such as x or eggs_sold. Write the numbers as the "
    "problem gives them and do not work anything out. Write * for times, / for divide, ^ for powers and "
    "sqrt( ) for square roots. On the last line, write \"Answer:\" and the name or expression the question "
    "asks for."
)
#: The reminder after each question, and worked examples given as earlier turns of the chat: a small model
#: follows a format it has seen answered more often than one it is only told about. The examples are written
#: for this file and are not from any benchmark.
FORMAL_ASK = "Write the equations and the Answer line. Do not work them out."
FORMAL_EXAMPLES = (
    ("A jacket costs $80. It is on sale for 25% off, and Sam also uses a $5 coupon. How much does Sam pay?",
     "price = 80\ndiscount = price * 25 / 100\npaid = price - discount - 5\nAnswer: paid"),
    ("Mia has twice as many stickers as Leo. Together they have 36 stickers. How many stickers does Mia "
     "have?",
     "mia = 2 * leo\nmia + leo = 36\nAnswer: mia"),
    ("A tap fills a 120-litre tank at 8 litres a minute. The tank already holds 40 litres. How many minutes "
     "does it take to fill the tank?",
     "missing = 120 - 40\nminutes = missing / 8\nAnswer: minutes"),
    ("If 3x + 7 = 22, what is the value of 2x - 1?", "3*x + 7 = 22\nAnswer: 2*x - 1"),
)
#: The ``json`` condition: the same instruction and examples, with the reply held to this JSON schema by the
#: model server, so that every reply has equations and an answer in fixed places.
JSON_ASK = ("Write the equations and the name of the quantity the question asks for as JSON, with "
            "\"equations\" a list of equations and \"answer\" the name. Do not work them out.")
JSON_FORMAT = {"type": "object", "required": ["equations", "answer"],
               "properties": {"equations": {"type": "array", "items": {"type": "string"}},
                              "answer": {"type": "string"}}}


def json_messages(question: str) -> list[dict]:
    """The chat for ``json``: the instruction, the worked examples written as JSON, then the question."""
    messages = [{"role": "system", "content": FORMAL_PROMPT}]
    for asked, answered in FORMAL_EXAMPLES:
        *equations, answer = answered.split("\n")
        reply = {"equations": equations, "answer": answer.removeprefix("Answer:").strip()}
        messages += [{"role": "user", "content": f"{asked}\n\n{JSON_ASK}"},
                     {"role": "assistant", "content": json.dumps(reply)}]
    return messages + [{"role": "user", "content": f"{question}\n\n{JSON_ASK}"}]


def from_json(reply: str) -> str:
    """A JSON reply as the lines ADA reads: the equations, then the Answer line. A reply that is not such JSON
    is read as it is."""
    try:
        found = json.loads(reply)
        equations = [str(line) for line in found["equations"]]
        answer = str(found["answer"])
    except (ValueError, TypeError, KeyError):
        return reply
    return "\n".join([*equations, f"Answer: {answer}"])


#: The same examples worked out step by step, for the ``shots`` condition: the model alone with as many
#: examples as ``solve`` has, so that what ``solve`` gains is not only from seeing examples.
WORKED_ANSWERS = (
    "The discount is 25% of $80, which is 80 * 25 / 100 = 20 dollars. Sam pays 80 - 20 - 5 = 55 dollars.\n"
    "Answer: 55",
    "Together they have Leo's stickers three times over, so Leo has 36 / 3 = 12 stickers and Mia has "
    "2 * 12 = 24.\nAnswer: 24",
    "The tank is missing 120 - 40 = 80 litres. At 8 litres a minute, that takes 80 / 8 = 10 minutes.\n"
    "Answer: 10",
    "3x = 22 - 7 = 15, so x = 15 / 3 = 5. Then 2x - 1 = 2 * 5 - 1 = 9.\nAnswer: 9",
)


#: The ``plan`` condition: the model works the problem out in words first, then writes the equations ADA
#: solves. The examples reason about what to work out, not the numbers, and their equations are those of
#: FORMAL_EXAMPLES.
PLAN_PROMPT = (
    "Work out in words how to solve the problem: which quantities it gives, which it asks for, and how they "
    "are related. Then write the problem as equations that an exact solver will solve. Put one equation on "
    "each line, and give each quantity a name of one word, such as x or eggs_sold. Write the numbers as the "
    "problem gives them and do not work anything out. Write * for times, / for divide, ^ for powers and "
    "sqrt( ) for square roots. On the last line, write \"Answer:\" and the name or expression the question "
    "asks for."
)
PLAN_ASK = "Work out how to solve it in words, then write the equations and the Answer line."
PLAN_REASONS = (
    "Sam pays the price less the discount and less the coupon. The discount is 25% of the price.",
    "Mia has twice as many as Leo, and the two together have 36: two unknowns and two equations.",
    "The tap has to fill the litres the tank is missing, which are its size less what it holds. At 8 litres "
    "a minute, the minutes are the missing litres divided by 8.",
    "The equation gives x, and the question asks for 2x - 1.",
)


def plan_messages(question: str) -> list[dict]:
    """The chat for ``plan``: the instruction, the examples reasoned out and written as equations, then the
    question."""
    messages = [{"role": "system", "content": PLAN_PROMPT}]
    for (asked, answered), reason in zip(FORMAL_EXAMPLES, PLAN_REASONS, strict=True):
        messages += [{"role": "user", "content": f"{asked}\n\n{PLAN_ASK}"},
                     {"role": "assistant", "content": f"{reason}\n\n{answered}"}]
    return messages + [{"role": "user", "content": f"{question}\n\n{PLAN_ASK}"}]


def missing(found: dict) -> list[str]:
    """The quantities ADA could find no number for, by the names the model gave them: first those no
    equation defines (``b`` in ``blue = 2*b``), and when every quantity is defined, those left without a
    number."""
    read, aim = found.get("read") or [], found.get("aim") or ""
    sets = _solutions(read)
    if isinstance(sets, Refusal):
        return []
    numbers = {name for values in sets for name, value in values.items() if _sympy(value) is not None}
    defined = {equation.split("=", 1)[0].strip() for equation in read}
    used = []
    for text in [*read, aim]:
        for name in _NAME.findall(text):
            if (name not in numbers and name not in used and name.lower() not in FUNCTION_NAMES
                    and name not in CONSTANTS):
                used.append(name)
    letters = {letter: name for name, letter in found.get("names", {}).items()}
    undefined = [name for name in used if name not in defined]
    return [letters.get(name, name) for name in (undefined or used)]


def feedback(found: dict) -> str | None:
    """What to tell the model when ADA found no one value for its equations, or the Answer line gives a number
    the model worked out instead of a quantity of its equations, so it can write them again; None when ADA
    solved the equations."""
    status = found["status"]
    if status == "solved":
        return None
    target = found.get("target")
    if status == "number given":
        reason = (f"Your Answer line gives a number you worked out, {target}. ADA works the numbers out: "
                  "write the equations with the numbers the problem gives, and after \"Answer:\" write the "
                  "name of the quantity the question asks for.")
    elif status == "no answer line":
        reason = "Your reply has no line that starts with \"Answer:\"."
    elif status == "example repeated":
        reason = "Your equations are those of an earlier example, not of this problem."
    elif status == "refused":
        reason = f"ADA could not solve your equations: {found.get('message', '')}"
    elif status == "several values":
        reason = (f"Your equations allow more than one value for {target}: "
                  f"{', '.join(found.get('values', []))}. Add what the problem says that picks one.")
    else:
        names = missing(found)
        reason = (f"ADA found no value for {target}. "
                  + (f"These quantities have no value: {', '.join(names)}. " if names else "")
                  + "Each quantity needs a number from the problem or an equation that defines it.")
    return (f"{reason} Write all the equations for this problem again, one on each line, then the Answer "
            "line. Do not work them out.")


def reading_feedback(found: dict, left_out: tuple[str, ...] | list[str]) -> str | None:
    """What ``pair`` tells the model: when ADA found no one value, ``feedback``'s reason with the problem's
    numbers the equations leave out; when ADA solved equations that leave out a number of the problem, those
    numbers; None when ADA solved equations that use every number."""
    advice = feedback(found)
    if not left_out:
        return advice
    named = left_out[0] if len(left_out) == 1 else ", ".join(left_out[:-1]) + " and " + left_out[-1]
    each = "it stands" if len(left_out) == 1 else "each stands"
    note = (f"Your equations leave out {named} from the problem. Each number in a problem like this one is "
            f"there for a reason: read the problem again and work out what {each} for.")
    if advice:
        return f"{note} {advice}"
    return (f"ADA solved your equations, but they leave out {named} from the problem. Each number in a "
            f"problem like this one is there for a reason: read the problem again and work out what {each} "
            "for. Write all the equations for this problem again, one on each line, then the Answer line. Do "
            "not work them out.")


def worked_turns(boxed: bool = False) -> list[dict]:
    """The example questions and their step-by-step answers, as earlier turns of a chat. With ``boxed``, each
    answer ends with the number in \\boxed{}, as MATH's solutions do, rather than an "Answer:" line."""
    turns = []
    for (asked, _), answered in zip(FORMAL_EXAMPLES, WORKED_ANSWERS, strict=True):
        if boxed:
            working, number = answered.rsplit("\nAnswer: ", 1)
            answered = f"{working} The answer is \\boxed{{{number}}}."
        turns += [{"role": "user", "content": asked}, {"role": "assistant", "content": answered}]
    return turns


def formal_messages(question: str) -> list[dict]:
    """The chat that asks a model to write a problem as equations: the instruction, the worked examples as
    earlier turns, then the question."""
    messages = [{"role": "system", "content": FORMAL_PROMPT}]
    for asked, answered in FORMAL_EXAMPLES:
        messages += [{"role": "user", "content": f"{asked}\n\n{FORMAL_ASK}"},
                     {"role": "assistant", "content": answered}]
    return messages + [{"role": "user", "content": f"{question}\n\n{FORMAL_ASK}"}]


_ANSWER = re.compile(r"^\W*answer\W*?:\s*(.*)$", re.IGNORECASE)
_NAME = re.compile(r"(?<![\\A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*")
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_DELIMITERS = re.compile(r"\$\$?|\\[()\[\]]|`+|\*\*")
_BOXED = re.compile(r"\\boxed\s*\{(.*)\}")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
#: Words in brackets after a value: "x = 5 (eggs)", "y = 3 (3 a day)".
_ASIDE = re.compile(r"\s*\((?=[^()]*[A-Za-z]{2})[A-Za-z0-9\s]+\)\s*$")
#: Letters a quantity may be given: not e (ADA's constant) or i.
_LETTERS = [letter for letter in string.ascii_lowercase + string.ascii_uppercase if letter not in "eiE"]


def written(text: str, last: bool = False) -> tuple[list[str], str | None]:
    """The equations a reply writes, one a line, and the quantity its "Answer:" line asks for. With ``last``,
    only the equations after the reply's last line of prose: a model that thinks aloud before it writes the
    equations, and tries some on the way, puts the ones it settles on at the end.

    A line is read as LaTeX or plain text: ``\\text{eggs sold}`` is the name ``eggs_sold``, ``\\times`` is *,
    a unit after a number is left out, and so are "let" and a label before an equation. What the model works
    out itself is not read: in ``white = 2 / 2 = 1`` the name is defined by what follows it, a line that
    defines a name again is left out, and so is arithmetic with no name in it. A sentence with an equation in
    it is not read."""
    equations, target, defined, answered = [], None, set(), []
    for raw in text.splitlines():
        line = _clean(_BULLET.sub("", _DELIMITERS.sub("", raw)).strip())
        found = _ANSWER.match(line)
        if found:
            target, answered = _target(found.group(1)), list(equations)
            continue
        equation = _equation(line)
        if equation is None:
            if line and last:
                # Prose: what comes before it was thinking aloud.
                equations, defined = [], set()
            continue
        left = equation.split(" = ", 1)[0]
        if _key(left) in defined:
            continue
        if equation not in equations:
            equations.append(equation)
        if _NAME.fullmatch(left):
            defined.add(_key(left))
    # With last, the equations just before the last Answer line, whatever prose follows it.
    return (answered if last else equations), target


def _equation(line: str) -> str | None:
    """A cleaned line as "left = right", or None when it is not an equation ADA reads."""
    if "=" not in line or re.search(r"[<>!≈≠?]|==", line):
        return None
    line = _LABEL.sub("", _ASIDE.sub("", line.rstrip(".,;")).rstrip(".,;"))
    left, right = (_unit_dropped(side.strip()) for side in line.split("=")[:2])
    if not left or not right or any(_PROSE.search(side) for side in (left, right)):
        return None
    if not _NAME.search(left + " " + right):
        return None
    return f"{left} = {right}"


#: Two words in a row: a sentence, not an equation.
_PROSE = re.compile(r"(?<![\\A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*\s+[A-Za-z_][A-Za-z0-9_]*")
#: A label before an equation: "Step 2: total = a + b".
_LABEL = re.compile(r"^[A-Za-z][\w ]*:\s*")
#: Words after a number at the end of a side: "180 miles".
_UNIT = re.compile(r"(?<=[\d)])((?:\s+[A-Za-z]{2,})+)$")
_WRITTEN = (("\\_", "_"), ("\\%", "%"), ("\\times", "*"), ("\\cdot", "*"), ("\\div", "/"), ("×", "*"),
            ("÷", "/"), ("−", "-"), ("'", ""), ("’", ""))


def _clean(line: str) -> str:
    """A line with LaTeX names and signs written plainly."""
    line = re.sub(r"\\q?quad\b.*$", "", line)
    line = re.sub(r"(?<=\d)\s*\\(?:text|mathrm)\s*\{[^{}]*\}", "", line)
    line = re.sub(r"\\(?:text|mathrm|textit|operatorname)\s*\{([^{}]*)\}",
                  lambda found: "_".join(found.group(1).replace("\\_", "_").split()), line)
    for written_as, plain in _WRITTEN:
        line = line.replace(written_as, plain)
    # 80,000 is a number, but gcd(12,180) is a function of two.
    line = re.sub(r"(?<![\d,])(?<![A-Za-z]\()\d{1,3}(?:,\d{3})+(?![\d,])",
                  lambda found: found.group().replace(",", ""), line)
    return re.sub(r"^(?:let|so|then|and|thus|where|hence)\s+", "", line, flags=re.IGNORECASE).strip()


def _unit_dropped(side: str) -> str:
    """A side without the unit after its last number, unless the words are functions or constants."""
    found = _UNIT.search(side)
    words = found.group(1).split() if found else []
    if found and not any(word.lower() in FUNCTION_NAMES or word in CONSTANTS for word in words):
        return side[:found.start()].strip()
    return side


def _key(name: str) -> str:
    """A name as one quantity: "Total" and "total" are the same, x and X are not."""
    return name.lower() if len(name) > 1 else name


def _target(text: str) -> str | None:
    text = _clean(text).strip().rstrip(".")
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
            name = _key(found.group())
            reserved = name in FUNCTION_NAMES or name in CONSTANTS and name != "e"
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
        return _NAME.sub(lambda found: mapping.get(_key(found.group()), found.group()), text)

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
    with several values takes the one that is above zero, when only one is.

    The equations after the reply's last line of prose are read first, and every equation in the reply when
    those have no one value. Equations that are a worked example's own are the model repeating the example,
    and have no value. Anything else that goes wrong in reading a reply is recorded as an error with its
    message, so one reply cannot stop a run."""
    try:
        return _read(text, positive)
    except Exception as error:  # noqa: BLE001 - a benchmark run records the failure and goes on
        return {"status": "error", "value": None, "message": f"{type(error).__name__}: {error}"[:500]}


def _read(text: str, positive: bool) -> dict:
    last, every = written(text, last=True), written(text)
    if _example(*last) or _example(*every):
        return {"status": "example repeated", "value": None, "equations": last[0], "target": last[1]}
    if last[0] != every[0]:
        found = _solved(*last, positive)
        if found["status"] == "solved":
            return {**found, "read_from": "the last equations"}
    return _solved(*every, positive)


def _example(equations: list[str], target: str | None) -> bool:
    """Whether the equations and target are those of one of the worked examples."""
    return bool(target) and any(written(answered) == (equations, target) for _, answered in FORMAL_EXAMPLES)


def _solved(equations: list[str], target: str | None, positive: bool) -> dict:
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


#: The most digits an exact number in the solve condition may have: Python writes out whole numbers of up to
#: 4,300 digits.
MAX_DIGITS = 4_000


def _sympy(expression: str):
    """The exact value of an expression with no unknowns left, with SymPy, or None. A power or a value with
    more than MAX_DIGITS digits has none: 2^235423523 is not worked out."""
    import sympy  # noqa: PLC0415 - only the solve condition needs it
    from sympy.parsing.sympy_parser import (  # noqa: PLC0415
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    text = expression.replace("^", "**")
    options = {"local_dict": {"e": sympy.E, "pi": sympy.pi},
               "transformations": standard_transformations + (implicit_multiplication_application,)}
    try:
        if _huge(parse_expr(text, evaluate=False, **options)):
            return None
        value = sympy.nsimplify(parse_expr(text, **options), rational=False)
        if value.free_symbols or not value.is_number or value.has(sympy.zoo, sympy.nan, sympy.oo):
            return None
        value = sympy.simplify(value)
        return None if _huge(value) else value
    except Exception:  # noqa: BLE001 - SymPy's parser raises many kinds of error on text it cannot read
        return None


def _huge(tree) -> bool:
    """Whether an expression holds a whole-number power or a fraction with more than MAX_DIGITS digits."""
    import sympy  # noqa: PLC0415

    for node in sympy.preorder_traversal(tree):
        if isinstance(node, sympy.Rational) and max(abs(node.p), node.q).bit_length() > MAX_DIGITS * 3.33:
            return True
        if isinstance(node, sympy.Pow) and node.exp.is_Integer and node.base.is_Rational:
            size = max(abs(node.base.p), node.base.q)
            if size > 1 and abs(int(node.exp)) * math.log10(size) > MAX_DIGITS:
                return True
    return False


def plain_number(value: str) -> str | None:
    """ADA's value as a plain decimal number: a fraction or a root is written to 10 decimal places, in digits
    only, never as 1.2e-05, since the answer is read digit by digit."""
    import sympy  # noqa: PLC0415 - only the solve condition needs it

    if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        return value
    number = _sympy(value)
    if number is None or not number.is_real:
        return None
    digits = Decimal(str(sympy.N(number, 60)))
    with localcontext() as context:
        # Room for every digit before the point and ten after it, however large the number is.
        context.prec = max(digits.adjusted(), 0) + 20
        rounded = digits.quantize(Decimal("1e-10"))
    text = format(rounded, "f").rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def latex_value(value: str) -> str:
    """ADA's value in LaTeX, for a grader that reads it: 3*sqrt(3) is 3 \\sqrt{3}."""
    import sympy  # noqa: PLC0415 - only the solve condition needs it

    number = _sympy(value)
    return value if number is None else sympy.latex(number)
