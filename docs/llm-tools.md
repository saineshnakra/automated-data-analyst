# ADA as the math tool of a language model

A language model writes numbers it never computed: an average from a glance
at a few rows, a percentage worked out in its head, a trend it expects to see.
ADA is the part that does the math. The model asks in plain words; ADA
computes the answer with pandas, SciPy and statsmodels and returns the
number with its calculation and the rows behind it, or refuses with the
reason. The model then reports a number it can show the working for.

This works with any model that can call a tool: open-source models run with
Ollama, LM Studio, llama.cpp or vLLM, and Claude, GPT and the coding
assistants built on them. A small local model gets the most from it: it no
longer does arithmetic at all, only picks the tool and reads the answer.

```bash
pip install automated-data-analyst
```

## Pick your setup

| You run | Do this |
|---|---|
| Ollama, in a terminal | `ada chat data.csv` ([below](#chat-in-the-terminal)) |
| LM Studio, Jan, Msty, Cherry Studio, AnythingLLM | add ADA as an MCP server ([below](#apps-that-take-mcp-servers)) |
| Open WebUI | `ada mcp --http`, then add it under External Tools ([below](#open-webui)) |
| Claude Desktop, Cursor, Windsurf, Cline, Continue | add ADA as an MCP server ([below](#apps-that-take-mcp-servers)) |
| VS Code | `.vscode/mcp.json` ([below](#vs-code)) |
| Claude Code, Codex | `ada skill`, or ADA as an MCP server ([below](#claude-code-and-codex)) |
| ollmcp, oterm or another terminal client for Ollama | add ADA as an MCP server in its config |
| your own code | `ada.tools`, or `ada.chat.chat()` ([below](#in-your-own-code)) |
| LangChain or LangGraph | `ada.tools.langchain()` ([below](#langchain-and-langgraph)) |

The model has to support tool calling. On Ollama those are the models tagged
"tools", such as qwen3, llama3.1, llama3.2, mistral and granite; in LM Studio,
the models with the hammer icon. See [Which local models work](#which-local-models-work).

## Chat in the terminal

```bash
ollama pull qwen3:1.7b        # 1.4 GB; any model with tool calling works
ada chat penguins.csv         # or a folder of files
```

```console
$ ada chat penguins.csv
ADA chat: qwen3:1.7b at http://localhost:11434/v1, reading penguins.csv. Ask about the data; Ctrl-D or 'exit' leaves.
> Which species has the longest bills on average?
  ADA ada_ask(path="penguins.csv", question="average bill_length_mm by species")
    -> answered: Chinstrap has the highest average bill_length_mm of any species, at 48.83 (n = 68). The species groups do differ (Welch ANOVA, p < 0.001; η² = 0.71). Based on the 342 rows with bill_length_mm recorded, of 344 rows. Each: Chinstrap 48.83; Gentoo 47.50; Adelie 38.79.
The species with the longest average bill length is **Chinstrap**, with an average of **48.83 mm** (n = 68). This result is based on the mean calculation of `bill_length_mm` grouped by species, sorted in descending order. The data spans 342 rows, and the values are as follows:
- **Chinstrap**: 48.83 mm
- **Gentoo**: 47.50 mm
- **Adelie**: 38.79 mm

The p-value indicates significant differences between species (p < 0.001), confirming the observed trends.
```

Each tool call and what ADA returned are printed above the answer, so every
number can be traced. `ada chat penguins.csv "which species has the longest
bills?"` asks one question and stops; `--json` prints the answer and the
calls as JSON.

`--model` picks the model (default: `ADA_MODEL`, else the server's only
model, else `qwen3:1.7b`). `--base-url` points at another server with an
OpenAI-compatible API: `http://localhost:1234/v1` for LM Studio,
`http://localhost:8080/v1` for llama.cpp's server, `http://localhost:8000/v1`
for vLLM. The model is told which files it has and their columns, so it
names them right.

## Apps that take MCP servers

`ada mcp` serves ADA's tools over the Model Context Protocol. Most apps take
a server in the same JSON, in a file usually called `mcp.json` (LM Studio:
Program, Install, Edit mcp.json; Claude Desktop: Settings, Developer, Edit
Config; Cursor: `~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "ada": {
      "command": "uvx",
      "args": ["automated-data-analyst", "mcp", "--root", "~/data"]
    }
  }
}
```

`uvx` comes with [uv](https://docs.astral.sh/uv/) and installs ADA the first
time the app starts it. With ADA installed by pip instead, use
`"command": "ada", "args": ["mcp", "--root", "~/data"]`; an app that can't
find `ada` needs its full path (`which ada`, or `where ada` on Windows).

`--root` is the folder the tools may read, `~/data` here. Without it, ADA
reads the folder the app starts it in, or your home folder when that is the
top of the disk, as desktop apps start it. Then ask in the app's chat:
"in ~/data/sales.csv, what was the average order value by month?"

### Open WebUI

Open WebUI reaches MCP servers over HTTP:

```bash
ada mcp --http --root ~/data            # serves http://127.0.0.1:8000/mcp
```

In Open WebUI, open Admin Settings, External Tools, add a server, set the
type to MCP (Streamable HTTP) and the URL to `http://127.0.0.1:8000/mcp`.
When Open WebUI runs in Docker, start ADA with `--host 0.0.0.0` and use
`http://host.docker.internal:8000/mcp`; anything that can reach your
computer on that port can then read the tables under the root. Turn the tool
on in a chat, with a model that supports tool calling.

### VS Code

```json
{
  "servers": {
    "ada": {"type": "stdio", "command": "uvx", "args": ["automated-data-analyst", "mcp"]}
  }
}
```

in `.vscode/mcp.json` reads the workspace's files.

### Claude Code and Codex

```bash
ada skill                                               # the skill: the assistant runs the ada command
claude mcp add ada -- uvx automated-data-analyst mcp          # or the MCP server, for Claude Code
codex mcp add ada -- uvx automated-data-analyst mcp           # and for Codex
```

The [skill](../ada/skills/ada/SKILL.md) works with both, local models
included when the assistant runs on one.

## In your own code

One question, with the loop done for you:

```python
from ada.chat import chat

reply = chat("which species has the longest bills?", "penguins.csv", model="qwen3:1.7b")
print(reply.answer)
for call in reply.calls:
    print(call.name, call.arguments, call.result["status"])
```

`ada.chat.Chat(data, model=..., base_url=...)` keeps a conversation over
several questions. To run the loop yourself, `ada.tools.definitions()` gives
the four tools in the format OpenAI-compatible servers take, and
`ada.tools.run(name, arguments)` runs the one the model picks:

| Tool | What it does |
|---|---|
| `ada_ask` | computes an answer from a file or folder: a count, mean, share, change, ranking, group comparison, correlation or trend |
| `ada_check_claim` | checks a statement against the data: true, false or cannot tell, with the reason |
| `ada_scan` | lists the tables in a folder, their columns and the keys that join them |
| `ada_compute` | computes an arithmetic expression exactly, with each step: fractions stay exact, and an expression that reads two ways is refused with both readings (see [Exact arithmetic](reference/compute.md)). Given equations, linear ones, a polynomial equation in one unknown or a system of polynomial equations, it solves them and checks the answer (see [Algebra](reference/algebra.md)) |

All four only read; none writes a file.

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
    reply = client.chat.completions.create(model="qwen3:1.7b", messages=messages, tools=tools.definitions())
    message = reply.choices[0].message
    if not message.tool_calls:
        print(message.content)
        break
    messages.append(message.model_dump(exclude_none=True))
    for call in message.tool_calls:
        result = tools.run(call.function.name, call.function.arguments, root="data", brief=True)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
```

The `ollama` package takes the same definitions: `ollama.chat(model,
messages, tools=tools.definitions())`, then `tools.run(call.function.name,
call.function.arguments, root="data", brief=True)` for each call.
`ada tools` prints the definitions as JSON for any other client, and
`ada tools --style mcp` in the form an MCP server lists them.

`root="data"` lets the model name only files inside that folder; a path
outside it, or one that does not exist, comes back as `{"status": "error",
"message": ...}` naming the data files the model may use. `brief=True` keeps
what a model needs: the status, the answer or message, the calculation, the
file, the rows used, and any assumption, warning or question; a claim gets a
`verdict` of "true", "false" or "cannot tell". Without it, `run` returns the
whole record, with the reading, the evidence and the file's checksum.
`tools.SYSTEM_PROMPT` tells the model which tool to use when, to take every
number from them, and to pass on ADA's question when a result is unsettled.

### The Anthropic API

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

### LangChain and LangGraph

`tools.langchain()` gives the four tools as LangChain tools, with the same
names, descriptions and arguments as `tools.definitions()`. Each returns the
brief result as JSON text, as `tools.run(..., brief=True)` does. It needs
langchain-core:

```bash
pip install "automated-data-analyst[langchain]"
```

With LangGraph's prebuilt agent, and any chat model that calls tools:

```python
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from ada import tools

agent = create_react_agent(ChatOllama(model="qwen3:1.7b"), tools.langchain(root="data"),
                           prompt=tools.SYSTEM_PROMPT)
reply = agent.invoke({"messages": [("user", "What was the average departure delay by airport in flights.csv?")]})
print(reply["messages"][-1].content)
```

The same list goes into a `ToolNode` in a graph of your own, or into
`model.bind_tools(...)` in a LangChain loop, where `tool.invoke(call)` runs
each call the model makes and returns the `ToolMessage` to send back.
`root="data"` keeps the model to the files inside that folder, as with
`tools.run`.

## How a call goes

1. The app sends the question, ADA's instructions and the three tool
   definitions to the model.
2. The model answers with a tool call instead of a number:
   `ada_ask(path="penguins.csv", question="average bill_length_mm by species")`.
3. ADA reads the file with pandas, makes the calculation, and returns the
   number, the calculation and the rows used, or a refusal with its reason.
4. The result goes back to the model, which answers with ADA's number.

Small models slip in a few set ways, and ADA takes each in its stride where
only one reading is possible: a file named without its folder, from the top
of the disk or without its extension is the one file of that name under the
root; a call that leaves out the path, or calls it `file`, gets the file the
question names, or the only data file there is; a statement sent to `ada_ask`
("the average flipper length is 210") is checked as a claim, and a question
sent to `ada_check_claim` is answered. A brief answer ends with every group's
value, so a model asked for each one has them. In `ada chat`, a model that
writes a tool call as text has the call made, a call made twice gets its
first result with a nudge to answer, and the last round offers no tools.

## Which local models work

Five questions about [penguins.csv](https://github.com/allisonhorst/palmerpenguins)
asked through `ada chat` with Ollama 0.40 on four CPU cores and no GPU: the
average body mass of Gentoo penguins, whether "the average flipper length is
210 mm" is true, which species has the longest bills, how many penguins each
island has, and whether Adelie penguins have the longest bills. An answer
counts when it gives ADA's number or verdict.

| Model | Download | Right, of 5 | Typical time per question |
|---|---|---|---|
| qwen3:1.7b | 1.4 GB | 5 | 80 s |
| qwen3:0.6b | 523 MB | 5 | 65 s |
| qwen2.5:1.5b | 986 MB | 3 | 20 s |
| llama3.2:1b | 1.3 GB | 2 | 17 s |
| qwen2.5:0.5b | 397 MB | 2 | 7 s |

Every number ADA returned was right; the misses are the models': llama3.2
copied 5,076 as 5,067, qwen2.5 left "Gentoo" or "by species" out of the
question it passed on and answered for every penguin, and the smallest models
sometimes call no tool at all. Small models also vary from run to run: in an
earlier run qwen2.5:1.5b got 4 and llama3.2 got 1. qwen3 models, which think
before they answer, are the ones to start with; on a computer with a GPU they
answer in seconds. Without ADA's tools a model never sees the file, so any
number it gives about it is made up.

## Shell commands, for agents that run them

Claude Code, Codex and other coding agents can run the `ada` command
directly. `ada skill` installs the instructions that tell them to
([Agent skill](../ada/skills/ada/SKILL.md)).

```bash
ada ask flights.csv "average dep_delay by origin" --json
ada claim flights.csv "The average departure delay was about 13 minutes." --json   # exit 0 verified, 1 failed check
ada scan data/ --json
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
person. A claim that a group has the highest or lowest value ("Adelie has the
highest average bill length") is checked against every other group too. A
refusal (`"status": "refused"`) has a `code` and a `message` that says why,
such as a column the data do not have.

Every number in a result comes from pandas, SciPy or statsmodels, so what the
model reports can be checked against the data and the calculation it names.
See [Checking claims](reference/claims.md) and [Ask ADA](reference/ask-ada.md).
