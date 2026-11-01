"""ADA as tools a language model calls: function definitions and a runner.

    from ada import tools
    tools.definitions()                # function-calling definitions, OpenAI style (also Ollama, vLLM,
                                       # llama.cpp's server and LM Studio)
    tools.definitions("anthropic")     # the same for the Anthropic Messages API
    tools.run("ada_check_claim", {"path": "flights.csv", "claim": "The average delay was 13 minutes."})

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
    "You answer questions about data files (CSV, TSV, Excel) with ADA's tools. Never compute, estimate or "
    "make up a number yourself.\n"
    "- To find a number (a count, average, total, share, change, ranking, group comparison, correlation or "
    "trend), call ada_ask with the file's path and the user's question whole, keeping every value and group "
    "it names.\n"
    "- When the user gives a statement to check, call ada_check_claim with the file's path and the "
    "statement.\n"
    "- When you do not know which file or column to use, call ada_scan.\n"
    "- Call ada_model only when asked to predict a column for new rows.\n"
    "Answer with the numbers the tool returns and its calculation, and work out no new ones from them (no "
    "sums, differences or percentages): ask the tool. When a tool refuses or cannot tell, say so, give its "
    "reason and ask the user the question it gives."
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
        "name": "ada_model",
        "title": "Predict a column",
        "description": (
            "Train cross-validated models (gradient boosting and a linear model) to predict one column of a "
            "table for new rows, as a tabular Kaggle competition asks. Use it only when asked to predict "
            "values for new rows, never for counts, averages or other questions about the data. Returns the "
            "cross-validated score, how it was computed, the features and the checks on the predictions. "
            "Needs automated-data-analyst[model]."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "train": {"type": "string", "description": "The training table, or a folder holding "
                                                           "train.csv, test.csv and a sample submission."},
                "test": {"type": "string", "description": "The table to predict for, when train is a file."},
                "target": {"type": "string", "description": "The column to predict (default: the one the "
                                                            "test table lacks)."},
                "metric": {"type": "string", "description": "accuracy, balanced_accuracy, f1, auc, "
                                                            "logloss, map@3, rmse, rmsle, mae or spearman."},
                "explain": {"type": "boolean", "description": "Predict with a model a person can read, "
                                                              "with its coefficients and effect curves."},
                "out": {"type": "string", "description": "Write the predictions to this CSV file."},
            },
            "required": ["train"],
        },
    },
)

#: The tools' names.
NAMES = tuple(tool["name"] for tool in _TOOLS)
#: The tools that only read: every one but ada_model, which may write its predictions.
READ_ONLY = frozenset({"ada_ask", "ada_check_claim", "ada_scan"})


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
        if isinstance(inner, dict) and not given.keys() & {"path", "question", "claim", "train"}:
            # {"type": "function", "parameters": {...}}: the arguments wrapped as the call itself.
            given = dict(inner)
        if not given.get("path") and (alias := next((key for key in PATH_ALIASES if given.get(key)), None)):
            # {"file": "penguins.csv"}: another name small models give the path.
            given["path"] = given[alias]
        if name == "ada_ask":
            question = _text(given, "question")
            result = _ask(_inside(given.get("path"), base, "path", text=question), question)
        elif name == "ada_check_claim":
            claim = _text(given, "claim")
            result = _claim(_inside(given.get("path"), base, "path", text=claim), claim)
        elif name == "ada_scan":
            from .folder import scan  # noqa: PLC0415

            result = scan(_inside(given.get("path"), base, "path")).to_dict()
        elif name == "ada_model":
            result = _model(given, base)
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


def _model(given: dict, base: Path | None):
    from .api import Refusal  # noqa: PLC0415
    from .modelling import model  # noqa: PLC0415

    train = _inside(given.get("train"), base, "train")
    test = _inside(given["test"], base, "test") if given.get("test") else None
    out = _inside(given["out"], base, "out", must_exist=False) if given.get("out") else None
    report = model(train, test, target=given.get("target"), metric=given.get("metric"),
                   explain=bool(given.get("explain")))
    if isinstance(report, Refusal):
        return report
    result = report.to_dict()
    if out is not None:
        report.write(out)
        result["written"] = str(out)
    return result


def _shown(value: Any) -> str:
    """A group's value as the sentence of a brief result shows it: 168, 48.83."""
    if not isinstance(value, (int, float)):
        return str(value)
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


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
    values = (result.get("result") or {}).get("values")
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
    if "score" in result:
        out.update({key: result[key] for key in ("target", "task", "metric", "score", "spread", "rows")
                    if key in result})
        out["importances"] = (result.get("importances") or [])[:5]
        out["failed_checks"] = [item for item in result.get("checks") or [] if not item.get("passed")]
    return out
