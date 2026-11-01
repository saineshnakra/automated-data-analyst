"""GSM8K with and without ADA: does a small local model get more grade-school word problems right when it
sends its arithmetic to ``ada_compute``?

    curl -L -o test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
    ollama pull qwen3:1.7b
    python benchmarks/gsm8k.py test.jsonl --model qwen3:1.7b --problems 100 --out results.jsonl

Each problem is asked twice with the same model, settings and prompt: once with no tools, once with
``ada_compute`` offered. The model ends its reply with "Answer: <number>", and that number is compared with
the answer after "####" in the data. Temperature is 0 and thinking is off, so a run repeats.

It talks to Ollama's own API (``/api/chat``) on this computer, with the standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ada import tools  # noqa: E402

PROMPT = (
    "Solve the math word problem. Think step by step, then end your reply with one line: "
    "Answer: <the final number>."
)
TOOL_PROMPT = (
    PROMPT + " Do no arithmetic in your head: for every calculation, call ada_compute with an expression "
    "made of the numbers, such as (16 - 3 - 4) * 2, and use the value it returns."
)
#: Tool-calling rounds before the model must answer.
ROUNDS = 8
ANSWER = re.compile(r"answer\s*[:=]?\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)
NUMBER = re.compile(r"-?[\d,]*\.?\d+")


def compute_tool() -> dict:
    """The ada_compute definition in the form Ollama's chat API takes."""
    return next(item for item in tools.definitions("openai") if item["function"]["name"] == "ada_compute")


def chat(base: str, model: str, messages: list[dict], offered: list[dict]) -> dict:
    body = {"model": model, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 0, "seed": 0, "num_ctx": 8192}}
    if offered:
        body["tools"] = offered
    request = urllib.request.Request(f"{base}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=900) as response:
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


def solve(base: str, model: str, question: str, with_ada: bool) -> dict:
    offered = [compute_tool()] if with_ada else []
    messages = [{"role": "system", "content": TOOL_PROMPT if with_ada else PROMPT},
                {"role": "user", "content": question}]
    calls = []
    for round_ in range(ROUNDS + 1):
        message = chat(base, model, messages, offered if round_ < ROUNDS else [])
        requested = message.get("tool_calls") or []
        messages.append({key: message[key] for key in ("role", "content", "tool_calls") if message.get(key)})
        if not requested:
            return {"reply": message.get("content", ""), "calls": calls}
        for call in requested:
            function = call.get("function") or {}
            result = tools.run(function.get("name", ""), function.get("arguments") or {}, brief=True)
            calls.append({"arguments": function.get("arguments"), "result": result})
            messages.append({"role": "tool", "content": json.dumps(result)})
    return {"reply": "", "calls": calls}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", help="GSM8K test.jsonl")
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
            done[(record["index"], record["with_ada"])] = record
    with out.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            gold = Fraction(row["answer"].split("####")[-1].strip().replace(",", ""))
            for with_ada in (False, True):
                if (index, with_ada) in done:
                    continue
                started = time.monotonic()
                solved = solve(args.base_url, args.model, row["question"], with_ada)
                given = final_number(solved["reply"])
                record = {"index": index, "with_ada": with_ada, "gold": str(gold),
                          "given": None if given is None else str(given), "correct": given == gold,
                          "seconds": round(time.monotonic() - started, 1), "tool_calls": len(solved["calls"]),
                          "reply": solved["reply"], "calls": solved["calls"]}
                done[(index, with_ada)] = record
                handle.write(json.dumps(record) + "\n")
                handle.flush()
    for with_ada in (False, True):
        records = [done[(index, with_ada)] for index in range(len(rows)) if (index, with_ada) in done]
        correct = sum(record["correct"] for record in records)
        label = "with ada_compute" if with_ada else "no tools"
        print(f"{args.model}, {label}: {correct}/{len(records)} correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
