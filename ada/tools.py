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
"""

import json
from pathlib import Path
from typing import Any

#: Instructions to give the model with the tools.
SYSTEM_PROMPT = (
    "You have ADA's tools for data files (CSV, TSV, Excel). Use them for every number about the data: "
    "do not compute or estimate numbers yourself. Report the number a tool returns with the calculation it "
    "gives. Before you state a number about the data, check it with ada_check_claim. When a result is "
    "unverified or refused, say so, give its reason, and ask the user the question it suggests instead of "
    "guessing."
)

_PATH = {"type": "string", "description": "A CSV, TSV or Excel file, or a folder of them."}

_TOOLS = (
    {
        "name": "ada_ask",
        "description": (
            "Compute an answer from a data file or a folder of them: a count, sum, mean, median, share, "
            "rate, change, ranking, group comparison, correlation or trend. Returns the numbers, the "
            "calculation, the rows used and left out, and any choice made; or a refusal with the reason."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": _PATH,
                "question": {"type": "string", "description": "The question in plain words, such as "
                                                              "'average dep_delay by origin'."},
            },
            "required": ["path", "question"],
        },
    },
    {
        "name": "ada_check_claim",
        "description": (
            "Check one number stated about a data file or folder against the data. Returns status "
            "'verified' or 'unverified' with a reason (failed_check means the data give a different "
            "number), the number ADA computed, the calculation and its assumptions, and a question to ask "
            "when the claim can be read more than one way."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": _PATH,
                "claim": {"type": "string", "description": "The sentence that states the number, such as "
                                                           "'The average departure delay was about 13 "
                                                           "minutes.'"},
            },
            "required": ["path", "claim"],
        },
    },
    {
        "name": "ada_scan",
        "description": (
            "Describe a folder of data files, or one file: each table's rows and columns, the keys that "
            "join the tables, and the questions to confirm before combining them."
        ),
        "parameters": {"type": "object", "properties": {"path": _PATH}, "required": ["path"]},
    },
    {
        "name": "ada_model",
        "description": (
            "Predict one column of a table from the others with cross-validated gradient boosting and a "
            "linear model, as a tabular Kaggle competition asks. Returns the cross-validated score, how it "
            "was computed, the features, the checks on the predictions and any warnings. Needs "
            "automated-data-analyst[model]."
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


def definitions(style: str = "openai") -> list[dict]:
    """The tools' definitions for function calling: ``"openai"`` (the default) or ``"anthropic"``.

    The OpenAI style is what OpenAI-compatible servers take, local ones
    included; the Anthropic style is the Messages API's ``tools``.
    """
    if style == "openai":
        return [{"type": "function", "function": json.loads(json.dumps(tool))} for tool in _TOOLS]
    if style == "anthropic":
        return [{"name": tool["name"], "description": tool["description"],
                 "input_schema": json.loads(json.dumps(tool["parameters"]))} for tool in _TOOLS]
    raise ValueError(f"style is 'openai' or 'anthropic', not {style!r}")


def _inside(value: Any, root: Path | None, name: str, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is required, as a path")
    path = Path(value)
    if root is not None:
        path = path if path.is_absolute() else root / path
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"{value} is outside the folder the tools may read")
    if must_exist and not path.exists():
        raise ValueError(f"{value} does not exist")
    return path


def _refused(refusal) -> dict:
    return {"status": "refused", "code": refusal.code, "message": refusal.message}


def run(name: str, arguments: dict | str, root: str | Path | None = None) -> dict:
    """Run one tool and return its result as JSON-safe data.

    ``arguments`` may be a dict or the JSON text a model sends. A problem with
    the call itself (an unknown tool, a missing or outside path) comes back as
    ``{"status": "error", "message": ...}``, for the model to read and retry.
    """
    from .api import Refusal  # noqa: PLC0415

    base = Path(root) if root is not None else None
    try:
        given = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        if name == "ada_ask":
            result = _ask(_inside(given.get("path"), base, "path"), _text(given, "question"))
        elif name == "ada_check_claim":
            from .claims import check_claim  # noqa: PLC0415

            result = check_claim(_inside(given.get("path"), base, "path"), _text(given, "claim")).to_dict()
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
    return json.loads(json.dumps(result, default=str))


def _text(given: dict, name: str) -> str:
    value = given.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required, as text")
    return value


def _ask(path: Path, question: str):
    from .api import Refusal, analyze  # noqa: PLC0415
    from .joins import ask_folder  # noqa: PLC0415

    answer = ask_folder(path, question) if path.is_dir() else analyze(path).ask(question)
    if isinstance(answer, Refusal):
        return {**_refused(answer), "question": question}
    return answer.to_dict()


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
