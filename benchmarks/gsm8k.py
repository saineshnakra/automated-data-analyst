"""GSM8K with and without ADA: does a small local model get more grade-school word problems right with ADA's
exact arithmetic?

    curl -L -o test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
    ollama pull qwen3:1.7b
    python benchmarks/gsm8k.py test.jsonl --model qwen3:1.7b --problems 100 --out results.jsonl

It reads GSM-Hard too (the same problems with larger numbers, ``gsmhardv2.jsonl`` from the PAL repository),
whose answers are floating-point results and are matched to within 0.001.

Each problem is solved three ways with the same model and settings:

- ``none``: the model alone.
- ``tool``: the model may call ``ada_compute`` for any arithmetic step. A call written out as text instead of
  made is made for it, as ``ada chat`` does.
- ``check``: the model alone, then ADA checks every equation it wrote (``ada.compute.check_arithmetic``).
  When one is wrong, the model is told which, with the right value, and answers again. When none is, the
  answer stands, so this condition never changes a reply ADA finds nothing wrong with.

The model ends its reply with a line "Answer: <number>", and that number is compared with the answer after
"####" in the data. Temperature is 0 and thinking is off, so a run repeats. It talks to Ollama's own API
(``/api/chat``) on this computer, with the standard library only.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import time
import urllib.request
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ada import tools  # noqa: E402
from ada.chat import _written_call  # noqa: E402
from ada.compute import check_arithmetic  # noqa: E402

PROMPT = (
    "Solve the math word problem. Think step by step. End your reply with a last line that starts with "
    "\"Answer:\" followed by the final number alone."
)
TOOL_PROMPT = (
    PROMPT + " You can call ada_compute to work out any arithmetic step exactly: write the expression with "
    "the numbers from your reasoning, and use the value it returns."
)
CONDITIONS = ("none", "tool", "check")
#: Tool-calling rounds before the model must answer.
ROUNDS = 8
#: The longest reply, in tokens: a small model sometimes repeats itself until it is stopped.
MAX_TOKENS = 2048
ANSWER = re.compile(r"answer\s*[:=]?\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)
NUMBER = re.compile(r"-?[\d,]*\.?\d+")


def compute_tool() -> dict:
    """The ada_compute definition in the form Ollama's chat API takes."""
    return next(item for item in tools.definitions("openai") if item["function"]["name"] == "ada_compute")


def chat(base: str, model: str, messages: list[dict], offered: list[dict], max_tokens: int = MAX_TOKENS,
         think: bool = False, options: dict | None = None) -> dict:
    """One reply from Ollama. ``options`` are added to the settings (temperature 0, seed 0)."""
    body = {"model": model, "messages": messages, "stream": False, "think": think,
            "options": {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": max_tokens,
                        **(options or {})}}
    if offered:
        body["tools"] = offered
    request = urllib.request.Request(f"{base}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=3600 if think else 900) as response:
        return json.loads(response.read())["message"]


def final_number(text: str) -> Fraction | None:
    """The number after the last "Answer:", else the last number in the reply."""
    found = ANSWER.findall(text)
    candidates = found or NUMBER.findall(text)
    if not candidates:
        return None
    try:
        return Fraction(candidates[-1].replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def solve(base: str, model: str, question: str, with_ada: bool,
          prompts: tuple[str, str] = (PROMPT, TOOL_PROMPT), max_tokens: int = MAX_TOKENS,
          think: bool = False, options: dict | None = None) -> dict:
    """The model's reply to one problem, with ada_compute offered when ``with_ada``; ``prompts`` are the
    system prompts without and with the tool, and ``think`` turns on a thinking model's thinking."""
    offered = [compute_tool()] if with_ada else []
    messages = [{"role": "system", "content": prompts[1] if with_ada else prompts[0]},
                {"role": "user", "content": question}]
    calls = []
    for round_ in range(ROUNDS + 1):
        message = chat(base, model, messages, offered if round_ < ROUNDS else [], max_tokens, think, options)
        requested = message.get("tool_calls") or []
        written = None if requested or not offered else _written_call(message.get("content", ""))
        if written is not None and written[0] == "ada_compute":
            # A small model sometimes writes the call out as text; ada chat makes it.
            requested = [{"function": {"name": written[0], "arguments": written[1]}}]
            message = {"role": "assistant", "content": "", "tool_calls": requested}
        messages.append({key: message[key] for key in ("role", "content", "tool_calls") if message.get(key)})
        if not requested:
            return {"reply": message.get("content", ""), "calls": calls, "messages": messages,
                    "thinking": message.get("thinking", "")}
        for call in requested:
            function = call.get("function") or {}
            result = tools.run(function.get("name", ""), function.get("arguments") or {}, brief=True)
            calls.append({"arguments": function.get("arguments"), "result": result})
            messages.append({"role": "tool", "content": json.dumps(result)})
    return {"reply": "", "calls": calls, "messages": messages}


def checked(base: str, model: str, alone: dict) -> dict:
    """The model's own answer, checked by ADA; when an equation in it is wrong, the model answers again."""
    checks = check_arithmetic(alone["reply"])
    # A model that repeats itself writes the same slip many times; it is named once.
    wrong = list({(item.expression, item.stated): item for item in checks if not item.correct}.values())
    if not wrong:
        return {"reply": alone["reply"], "calls": [], "checks": [item.to_dict() for item in checks]}
    errors = "; ".join(f"{item.expression} is {item.computed}, not {item.stated}" for item in wrong)
    messages = alone["messages"] + [{"role": "user", "content": (
        f"ADA checked the arithmetic in your answer and found errors: {errors}. Correct every step that uses "
        "these numbers, then end with a last line that starts with \"Answer:\" followed by the final number "
        "alone.")}]
    message = chat(base, model, messages, [])
    return {"reply": message.get("content", ""), "calls": [], "checks": [item.to_dict() for item in checks],
            "first_reply": alone["reply"]}


def problem(row: dict) -> tuple[str, Fraction, Fraction]:
    """The question, the answer and how far from it still counts as right, from a GSM8K or GSM-Hard row."""
    if "target" in row:
        return row["input"], Fraction(str(row["target"])), Fraction(1, 1000)
    return row["question"], Fraction(row["answer"].split("####")[-1].strip().replace(",", "")), Fraction(0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", help="GSM8K test.jsonl, or GSM-Hard gsmhardv2.jsonl")
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--problems", type=int, default=100, help="the first N problems (default: 100)")
    parser.add_argument("--out", required=True, help="write one JSON line per problem and condition here")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line]
    rows = rows[:args.problems]
    done = {}
    out = Path(args.out)
    if out.exists():
        # A run that stopped picks up where it left off.
        for line in out.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done[(record["index"], record["condition"])] = record
    with out.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            question, gold, tolerance = problem(row)
            alone = None
            for condition in CONDITIONS:
                if (index, condition) in done:
                    continue
                started = time.monotonic()
                try:
                    if condition == "check":
                        alone = alone or solve(args.base_url, args.model, question, False)
                        solved = checked(args.base_url, args.model, alone)
                    else:
                        solved = solve(args.base_url, args.model, question, condition == "tool")
                        if condition == "none":
                            alone = solved
                except (OSError, http.client.HTTPException) as error:
                    # A reply that never ends or a dropped connection counts as no answer.
                    solved = {"reply": "", "calls": [], "error": str(error)}
                given = final_number(solved["reply"])
                record = {"index": index, "condition": condition, "gold": str(gold),
                          "given": None if given is None else str(given),
                          "correct": given is not None and abs(given - gold) <= tolerance,
                          "seconds": round(time.monotonic() - started, 1), "tool_calls": len(solved["calls"]),
                          "reply": solved["reply"], "calls": solved["calls"],
                          **{key: solved[key] for key in ("checks", "first_reply", "error") if key in solved}}
                done[(index, condition)] = record
                handle.write(json.dumps(record) + "\n")
                handle.flush()
    for condition in CONDITIONS:
        records = [done[(index, condition)] for index in range(len(rows)) if (index, condition) in done]
        correct = sum(record["correct"] for record in records)
        print(f"{args.model}, {condition}: {correct}/{len(records)} correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
