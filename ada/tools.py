"""ADA as tools a language model calls: function definitions and a runner.

    from ada import tools
    tools.definitions()                # function-calling definitions, OpenAI style (also Ollama, vLLM,
                                       # llama.cpp's server and LM Studio)
    tools.definitions("anthropic")     # the same for the Anthropic Messages API
    tools.run("ada_check_claim", {"path": "flights.csv", "claim": "The average delay was 13 minutes."})
    tools.run("ada_compute", {"expression": "0.15 * (1200 - 300)"})
    tools.langchain(root="data")       # the same tools for LangChain agents and LangGraph

The model asks and ADA does the math: every number comes from pandas, SciPy
or statsmodels with its calculation, or comes back refused with the reason.
``run`` returns JSON-safe data to send back as the tool's result. With
``root``, a model may only name files inside that folder.

Small models make the same few slips, and ``run`` takes them in its stride
where only one reading is possible: a file named without its folder, a path
left out when the question names the file or the folder holds one table, a
statement sent as a question and a question sent as a claim.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

#: Instructions to give the model with the tools.
SYSTEM_PROMPT = (
    "You answer questions about numbers and data files (CSV, TSV, Excel) with ADA's tools. Never compute, "
    "estimate or make up a number yourself.\n"
    "- To find a number (a count, average, total, share, change, ranking, group comparison, correlation or "
    "trend), call ada_ask with the file's path and the user's question whole, keeping every value and group "
    "it names.\n"
    "- When the user gives a statement to check, call ada_check_claim with the file's path and the "
    "statement.\n"
    "- When you do not know which file or column to use, call ada_scan.\n"
    "- For arithmetic on numbers you have (from the user, the question or a tool's result), call ada_compute "
    "with one expression, such as 0.15*(1200 - 300) or 7/12*100, or with equations to solve, such as "
    "2x + 3 = 11.\n"
    "Answer with the numbers the tools return and their calculation, and work out no new ones in your "
    "head (no sums, differences or percentages): ask ada_compute. When a tool refuses or cannot tell, say "
    "so, give its reason and ask the user the question it gives."
)

_PATH = {"type": "string", "description": "The path of the data file (CSV, TSV or Excel), or of a folder of "
                                          "them."}

_TOOLS = (
    {
        "name": "ada_ask",
        "title": "Compute a number from data",
        "description": (
            "Compute a number from a data file: a count, sum, average, median, share, rate, change, ranking, "
            "group comparison, correlation or trend. Use it for every question about the data. Returns the "
            "answer with its calculation and the rows used, or a refusal with the reason."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": _PATH,
                "question": {"type": "string", "description": "The user's question in plain words, with the "
                                                              "column names where you know them."},
            },
            "required": ["path", "question"],
        },
    },
    {
        "name": "ada_check_claim",
        "title": "Check a stated number",
        "description": (
            "Check a statement someone made about a data file: a number it gives, or a group it says is the "
            "highest or lowest. Use it for a statement, not for a question. Returns a verdict (true, false "
            "or cannot tell), the number ADA computed and the calculation, and a question to ask when the "
            "statement can be read more than one way."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": _PATH,
                "claim": {"type": "string",
                          "description": "The statement to check, word for word as the user gave it."},
            },
            "required": ["path", "claim"],
        },
    },
    {
        "name": "ada_scan",
        "title": "List the tables in a folder",
        "description": (
            "List the tables in a folder or file: each table's rows and columns, and the keys that join "
            "them. Use it when you do not know which file or column to use."
        ),
        "parameters": {"type": "object", "properties": {"path": _PATH}, "required": ["path"]},
    },
    {
        "name": "ada_compute",
        "title": "Compute an expression or solve equations exactly",
        "description": (
            "Compute an arithmetic expression exactly and show each step: + - * / ^, parentheses, mod, "
            "percent (20% of 50), factorial (5!), sqrt, root, abs, floor, ceil, round, gcd, lcm, comb, perm, "
            "min, max, exp, ln, log10, log2, log(x, base), pi and e. Fractions stay exact (1/3 + 1/6 = 1/2). "
            "Use it for every calculation on numbers you already have. It also solves equations with "
            "one-letter unknowns, one or several separated by ; (x + y = 10; x - y = 2), when each unknown "
            "is to the first power, and checks the answer by putting it back. An expression that reads two "
            "ways is refused with both readings."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "One arithmetic expression, such as "
                                                                "(1200 - 300) * 0.15 or comb(10, 3) / 2^10, "
                                                                "or equations such as 3x - 5 = 2x + 7."},
                "digits": {"type": "integer", "description": "Significant digits for a result that is not "
                                                            "exact, such as sqrt(2) (default 20)."},
            },
            "required": ["expression"],
        },
    },
)

#: The tools' names.
NAMES = tuple(tool["name"] for tool in _TOOLS)
#: The tools that only read: all of them, since none writes a file.
READ_ONLY = frozenset({"ada_compute", "ada_ask", "ada_check_claim", "ada_scan"})


def definitions(style: str = "openai") -> list[dict]:
    """The tools' definitions for function calling: ``"openai"`` (the default), ``"anthropic"`` or ``"mcp"``.

    The OpenAI style is what OpenAI-compatible servers take, local ones
    included; the Anthropic style is the Messages API's ``tools``; the MCP style
    is a Model Context Protocol server's ``tools/list``.
    """
    if style == "openai":
        return [{"type": "function", "function": {key: json.loads(json.dumps(tool[key]))
                                                  for key in ("name", "description", "parameters")}}
                for tool in _TOOLS]
    if style == "anthropic":
        return [{"name": tool["name"], "description": tool["description"],
                 "input_schema": json.loads(json.dumps(tool["parameters"]))} for tool in _TOOLS]
    if style == "mcp":
        return [{"name": tool["name"], "title": tool["title"], "description": tool["description"],
                 "inputSchema": json.loads(json.dumps(tool["parameters"])),
                 "annotations": {"title": tool["title"], "readOnlyHint": tool["name"] in READ_ONLY,
                                 "openWorldHint": False}} for tool in _TOOLS]
    raise ValueError(f"style is 'openai', 'anthropic' or 'mcp', not {style!r}")


def langchain(root: str | Path | None = None) -> list:
    """ADA's tools as LangChain tools, for LangChain agents and for LangGraph's ``ToolNode`` and
    ``create_react_agent``.

    Each tool takes the same arguments as its definition and returns the brief result as JSON text, as
    ``run(name, arguments, root=root, brief=True)`` does. With ``root``, the model may only name files inside
    that folder. Needs langchain-core: ``pip install "automated-data-analyst[langchain]"``.
    """
    try:
        from langchain_core.tools import StructuredTool  # noqa: PLC0415
    except ImportError as error:
        raise ImportError('ADA\'s LangChain tools need langchain-core: '
                          'pip install "automated-data-analyst[langchain]"') from error

    def runner(name: str):
        def call(**arguments: Any) -> str:
            return json.dumps(run(name, arguments, root=root, brief=True))
        return call

    return [StructuredTool.from_function(func=runner(tool["name"]), name=tool["name"],
                                         description=tool["description"],
                                         args_schema=json.loads(json.dumps(tool["parameters"])))
            for tool in _TOOLS]


#: Data files a message lists at most, so a model that left out or misspelt a path can pick the right one.
LISTED_FILES = 20
#: How deep below the root, and through how many folders, ADA looks for data files to list or match: a root
#: such as a home folder holds far more than a model needs to see.
SEARCH_DEPTH, SEARCH_FOLDERS = 3, 2000
SKIPPED_FOLDERS = frozenset({"node_modules", "__pycache__", "site-packages", "venv", "env", "Library"})


def _data_file_paths(root: Path) -> tuple[list[Path], bool]:
    """The data files under ``root``, in a fixed order, and whether the search stopped before the end."""
    from .file_io import SUPPORTED_SUFFIXES  # noqa: PLC0415

    found, visited = [], 0
    for folder, folders, names in os.walk(root):
        depth = len(Path(folder).relative_to(root).parts)
        folders[:] = [] if depth >= SEARCH_DEPTH else sorted(
            item for item in folders if not item.startswith(".") and item not in SKIPPED_FOLDERS)
        found += [Path(folder, item) for item in sorted(names)
                  if not item.startswith(".") and Path(item).suffix.lower() in SUPPORTED_SUFFIXES]
        visited += 1
        if visited >= SEARCH_FOLDERS:
            return found, True
    return found, False


def _data_files(root: Path | None) -> str:
    """The sentence naming the data files under ``root``, in a fixed order, for a model to pick from."""
    if root is None or not root.is_dir():
        return ""
    paths, cut = _data_file_paths(root)
    found = [path.relative_to(root).as_posix() for path in paths]
    if not found:
        return " There are no data files in the folder the tools may read."
    shown = found[:LISTED_FILES]
    listed = shown[0] if len(shown) == 1 else f"{', '.join(shown[:-1])} and {shown[-1]}"
    more = ", among others" if len(found) > LISTED_FILES or cut else ""
    return f" The data files are {listed}{more}."


# A data file named in a question or claim: "in data/penguins.csv", "from 'sales.xlsx'".
NAMED_FILE = re.compile(r"[\w./\\~-]*\.(?:csv|tsv|txt|data|dat|xlsx|xlsm)\b", re.IGNORECASE)


def _guessed(root: Path, text: str) -> Path | None:
    """The one data file a call that left out its path can mean: the file its text names, or the only data
    file under ``root``."""
    paths, cut = _data_file_paths(root)
    for named in NAMED_FILE.findall(text):
        candidate = Path(named).expanduser()
        candidate = candidate if candidate.is_absolute() else root / candidate
        if candidate.is_file() and candidate.resolve().is_relative_to(root.resolve()):
            return candidate
        same = [path for path in paths if path.name.lower() == Path(named).name.lower()]
        if len(same) == 1:
            return same[0]
    return paths[0] if len(paths) == 1 and not cut else None


def _same_name(root: Path, path: Path) -> list[Path]:
    """The data files under ``root`` with the name of ``path``, or its stem when it has no extension:
    "penguins" for penguins.csv."""
    name = path.name.lower()
    return [item for item in _data_file_paths(root)[0]
            if item.name.lower() == name or (not path.suffix and item.stem.lower() == name)]


def _inside(value: Any, root: Path | None, name: str, must_exist: bool = True, text: str = "") -> Path:
    """The path a call names, inside ``root``; a path left out, or a file named without its folder, is
    taken when only one data file fits."""
    if (not isinstance(value, str) or not value) and must_exist and root is not None and root.is_dir():
        guessed = _guessed(root, text)
        if guessed is not None:
            return guessed
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is required: the path of a data file or folder.{_data_files(root)}")
    path = Path(value).expanduser()
    if root is not None:
        path = path if path.is_absolute() else root / path
        if not path.resolve().is_relative_to(root.resolve()):
            # "/penguins.csv" for penguins.csv under the root: the one file of that name there.
            same = _same_name(root, path) if must_exist and root.is_dir() else []
            if len(same) == 1:
                return same[0]
            raise ValueError(f"{value} is outside the folder the tools may read.{_data_files(root)}")
    if must_exist and not path.exists() and root is not None and root.is_dir():
        # "penguins.csv" for data/penguins.csv: the one file of that name under the root.
        same = _same_name(root, path)
        if len(same) == 1:
            return same[0]
    if must_exist and not path.exists():
        raise ValueError(f"{value} does not exist.{_data_files(root)}")
    return path


def _refused(refusal) -> dict:
    return {"status": "refused", "code": refusal.code, "message": refusal.message}


def run(name: str, arguments: dict | str, root: str | Path | None = None, brief: bool = False) -> dict:
    """Run one tool and return its result as JSON-safe data.

    ``arguments`` may be a dict or the JSON text a model sends. A problem with
    the call itself (an unknown tool, a missing or outside path) comes back as
    ``{"status": "error", "message": ...}``, for the model to read and retry.
    ``brief`` keeps what a model needs to answer (see :func:`brief`).
    """
    from .api import Refusal  # noqa: PLC0415

    base = Path(root).expanduser() if root is not None else None
    try:
        given = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        if not isinstance(given, dict):
            raise ValueError("the arguments are a JSON object, such as {\"path\": ..., \"question\": ...}")
        inner = given.get("parameters", given.get("arguments"))
        if isinstance(inner, dict) and not given.keys() & {"path", "question", "claim", "expression"}:
            # {"type": "function", "parameters": {...}}: the arguments wrapped as the call itself.
            given = dict(inner)
        if not given.get("path") and (alias := next((key for key in PATH_ALIASES if given.get(key)), None)):
            # {"file": "penguins.csv"}: another name small models give the path.
            given["path"] = given[alias]
        if name == "ada_compute":
            from .algebra import work_out  # noqa: PLC0415
            from .compute import DIGITS  # noqa: PLC0415

            if not given.get("expression") and (alias := next(
                    (key for key in EXPRESSION_ALIASES if isinstance(given.get(key), str)), None)):
                # {"expr": "2 + 2"}: another name models give the expression.
                given["expression"] = given[alias]
            digits = str(given.get("digits", DIGITS))
            digits = int(digits) if digits.isdigit() else DIGITS
            computed = work_out(_text(given, "expression"), digits=digits)
            result = computed if isinstance(computed, Refusal) else computed.to_dict()
        elif name == "ada_ask":
            question = _text(given, "question")
            result = _ask(_inside(given.get("path"), base, "path", text=question), question)
        elif name == "ada_check_claim":
            claim = _text(given, "claim")
            result = _claim(_inside(given.get("path"), base, "path", text=claim), claim)
        elif name == "ada_scan":
            from .folder import scan  # noqa: PLC0415

            result = scan(_inside(given.get("path"), base, "path")).to_dict()
        else:
            return {"status": "error",
                    "message": f"no tool called {name!r}; the tools are {', '.join(NAMES)}"}
    except (ValueError, TypeError, OSError) as error:
        return {"status": "error", "message": str(error)}
    if isinstance(result, Refusal):
        result = _refused(result)
    result = json.loads(json.dumps(result, default=str))
    return globals()["brief"](result) if brief else result


def _text(given: dict, name: str) -> str:
    value = given.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required, as text")
    return value


#: Other names models give the expression argument.
EXPRESSION_ALIASES = ("expr", "input", "formula", "calculation", "math", "query", "question", "text")
#: Other names models give the path argument.
PATH_ALIASES = ("file", "file_path", "filepath", "filename", "file_name", "data", "dataset", "table", "csv")

# A sentence that states a number or a ranking instead of asking for one: "the average flipper length is 210".
STATED = re.compile(
    r"\b(?:is|was|are|were|equals?|totals?|totalled|reached|stood\s+at|has|had|have)\s+(?:the\s+)?"
    r"(?:about\s+|around\s+|roughly\s+|nearly\s+|almost\s+|over\s+|under\s+|more\s+than\s+|less\s+than\s+)?"
    r"(?:[$€£]?-?\d|(?:highest|lowest|largest|smallest|most|fewest)\b)",
    re.IGNORECASE,
)


def _ask(path: Path, question: str):
    from .api import Refusal, analyze  # noqa: PLC0415
    from .joins import ask_folder  # noqa: PLC0415

    answer = ask_folder(path, question) if path.is_dir() else analyze(path).ask(question)
    if isinstance(answer, Refusal):
        if not path.is_dir() and STATED.search(question) and not question.rstrip().endswith("?"):
            # "the average flipper length is 210" states a number: checked as a claim when that settles it.
            claim = _claim(path, question, asked=False)
            if claim["status"] == "verified" or claim.get("reason") == "failed_check":
                return {**claim, "note": "This states a number instead of asking for one, so ADA checked it "
                                         "as a claim."}
        return {**_refused(answer), "question": question}
    return answer.to_dict()


def _claim(path: Path, claim: str, asked: bool = True) -> dict:
    from .claims import NO_NUMBER, check_claim  # noqa: PLC0415

    check = check_claim(path, claim).to_dict()
    if asked and check.get("message") == NO_NUMBER and not path.is_dir():
        # "Which species has the longest bills?" asks for a number: answered when ADA can.
        answer = _ask(path, claim)
        if answer.get("status") == "answered":
            return {**answer, "note": "This asks for a number instead of stating one, so ADA answered it."}
    return check


def _shown(value: Any) -> str:
    """A group's value as the sentence of a brief result shows it: 168, 48.83."""
    if not isinstance(value, (int, float)):
        return str(value)
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


#: Steps of a computation a brief result keeps.
BRIEF_STEPS = 12
#: The verdict a brief claim result gives, in the words a person would use.
VERDICTS = {"verified": "true", "failed_check": "false"}


def brief(result: dict) -> dict:
    """A tool's result cut to what a model needs to answer: the status, the answer or message, the
    calculation, the file and rows used, and any assumption, warning or question.

    A claim's result gets a ``verdict`` of "true", "false" or "cannot tell",
    since "unverified" alone reads as "unknown" to a small model. The full result
    has the same numbers with the reading, the evidence and the file's checksum.
    """
    status = result.get("status")
    out: dict[str, Any] = {"status": status} if status else {}
    if "claim" in result and "agreement" in result:
        out["verdict"] = VERDICTS.get("verified" if status == "verified" else result.get("reason"),
                                      "cannot tell")
    for key in ("note", "claim", "question", "answer", "message", "reason", "code", "calculation"):
        if result.get(key):
            out[key] = result[key]
    evidence = result.get("evidence") or {}
    source = evidence.get("source") or {}
    if source.get("path"):
        out["file"] = source["path"]
    if evidence.get("rows_used") is not None:
        out["rows_used"] = evidence["rows_used"]
    inner = result.get("result")
    values = inner.get("values") if isinstance(inner, dict) else None
    if values:
        out["values"] = values[:LISTED_FILES]
        groups = [item for item in values if isinstance(item, dict) and "group" in item]
        if out.get("answer") and 1 < len(groups) <= LISTED_FILES and not all(
                str(item["group"]) in out["answer"] for item in groups):
            # "Biscoe has the most rows, at 168": every group in the sentence, so a model asked for each one
            # need not ask again.
            each = "; ".join(f"{item['group']} {_shown(item.get('value'))}" for item in groups)
            out["answer"] = f"{out['answer']} Each: {each}."
    for key in ("assumptions", "warnings", "clarification", "written"):
        if result.get(key):
            out[key] = result[key]
    if "files" in result:
        out["files"] = [{"path": (item.get("source") or {}).get("path") or item.get("name"),
                         "rows": item.get("rows"), "columns": item.get("columns"),
                         **({"problem": item["problem"]} if item.get("problem") else {})}
                        for item in result["files"]]
        out["links"] = result.get("links") or []
    if status == "solved":
        out.update({key: result[key] for key in ("kind", "values", "free", "conditions") if result.get(key)})
    if status == "simplified":
        out.update({key: result[key] for key in ("result", "conditions") if result.get(key)})
    if status in ("computed", "solved", "simplified"):
        out.update({key: result[key] for key in ("value", "exact") if key in result})
        steps = result.get("steps") or []
        if steps:
            out["steps"] = steps[:BRIEF_STEPS] + ([f"… and {len(steps) - BRIEF_STEPS} more steps"]
                                                  if len(steps) > BRIEF_STEPS else [])
    return out
