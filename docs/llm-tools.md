# ADA as the math tool of a language model

A language model writes numbers it never computed: an average from a glance
at a few rows, a percentage worked out in its head, a trend it expects to see.
ADA is the part that does the math. The model asks in plain words; ADA
computes the answer with pandas, SciPy and statsmodels and returns the
number with its calculation and the rows behind it, or refuses with the
reason. The model then reports a number it can show the working for.

This works with any model that can call a function or run a command:
Claude, GPT models, and open-source models run locally with Ollama,
llama.cpp, vLLM or LM Studio. A small local model gets the most from it: it
no longer has to do arithmetic at all, only pick the tool and read the
answer.

```bash
pip install automated-data-analyst
```

## Three ways in

| The model can | Use | Set up with |
|---|---|---|
| call functions (tool calling) | `ada.tools` | `tools.definitions()` and `tools.run(...)` |
| run shell commands | the `ada` command with `--json` | nothing; `ada skill` for Claude Code and Codex |
| run Python | `ada.check_claim`, `ada.analyze(...).ask`, `ada.scan`, `ada.model` | `import ada` |

## Tool calling, with a local open-source model

`ada.tools.definitions()` gives four tools in the format OpenAI-compatible
servers take, Ollama, llama.cpp's server, vLLM and LM Studio among them, and
`ada.tools.run(name, arguments)` runs the one the model picks:

| Tool | What it does |
|---|---|
| `ada_ask` | computes an answer from a file or folder: a count, mean, share, change, ranking, group comparison, correlation or trend |
| `ada_check_claim` | checks one stated number against the data: verified, or unverified with the reason |
| `ada_scan` | describes a folder of files and the keys that join them |
| `ada_model` | predicts a column with cross-validated models (needs `automated-data-analyst[model]`) |

A complete loop with Ollama, using the `openai` package (`pip install
openai`); any model Ollama serves with tool support works:

```python
import json

from openai import OpenAI

from ada import tools

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # Ollama's OpenAI-compatible API
messages = [
    {"role": "system", "content": tools.SYSTEM_PROMPT},
    {"role": "user", "content": "In flights.csv, what was the average departure delay by airport?"},
]
while True:
    reply = client.chat.completions.create(model="llama3.1", messages=messages, tools=tools.definitions())
    message = reply.choices[0].message
    if not message.tool_calls:
        print(message.content)
        break
    messages.append(message.model_dump(exclude_none=True))
    for call in message.tool_calls:
        result = tools.run(call.function.name, call.function.arguments, root="data")
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
```

For llama.cpp's server, vLLM or LM Studio, change `base_url` (for example
`http://localhost:8080/v1`, `http://localhost:8000/v1` or
`http://localhost:1234/v1`) and the model name. `ada tools` prints the same
definitions as JSON for any other client.

`root="data"` lets the model name only files inside that folder; a path
outside it, or a file that does not exist, comes back as
`{"status": "error", "message": ...}` for the model to read and correct.
`tools.SYSTEM_PROMPT` tells the model to take every number from the tools,
check the numbers it states, and pass on ADA's question when a result is
unverified.

## The Anthropic API

```python
import json

import anthropic

from ada import tools

client = anthropic.Anthropic()
messages = [{"role": "user", "content": "Is it true that the average price in sales.csv was 99?"}]
while True:
    reply = client.beta.messages.create(
        model="claude-opus-5-5", max_tokens=16000, system=tools.SYSTEM_PROMPT,
        tools=tools.definitions("anthropic"), messages=messages,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",  # another model takes a declined request
    )
    messages.append({"role": "assistant", "content": reply.content})
    if reply.stop_reason != "tool_use":
        print("".join(block.text for block in reply.content if block.type == "text"))
        break
    messages.append({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(tools.run(block.name, block.input))}
        for block in reply.content if block.type == "tool_use"]})
```

Every tool result of a turn goes back in one user message, as above.

## Shell commands, for agents that run them

Claude Code, Codex and other coding agents can run the `ada` command
directly. `ada skill` installs the instructions that tell them to
([Agent skill](../ada/skills/ada/SKILL.md)).

```bash
ada ask flights.csv "average dep_delay by origin" --json
ada claim flights.csv "The average departure delay was about 13 minutes." --json   # exit 0 verified, 1 failed check
ada scan data/ --json
ada model titanic/ --metric accuracy --out submission.csv --json
```

## What comes back

A computed answer (`ada_ask`):

```json
{"status": "answered", "question": "average price",
 "answer": "Average price is $98.94, calculated from 200 rows.",
 "result": {"value": 98.941, "figures": {"n": 200.0}},
 "reading": {"operation": "mean", "measure": "price"}, "calculation": "mean(price)"}
```

A checked claim (`ada_check_claim`): `status` is `verified`, or `unverified`
with a `reason`. Only `failed_check` says the data give a different number;
`missing_context` and `ambiguous_intent` come with a `question` to ask the
person. A refusal (`"status": "refused"`) has a `code` and a `message` that
says why, such as a column the data do not have.

Every number in a result comes from pandas, SciPy or statsmodels, so what the
model reports can be checked against the data and the calculation it names.
See [Checking claims](reference/claims.md), [Ask ADA](reference/ask-ada.md)
and [Predicting a column](reference/modelling.md).
