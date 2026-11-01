# Privacy

What ADA does with your data. To report a vulnerability, see
[SECURITY.md](../SECURITY.md).

## Summary

**Running ADA.** Nothing leaves the machine. Reading, cleaning, schema
detection, evidence cards, Ask ADA answers, claim checks and folder scans are
computed in-process with pandas, SciPy and statsmodels. ADA has no database,
needs no API key and makes no network call, except as below.

**`ada chat`.** Talks to the model server it is pointed at: Ollama on this
computer by default, or the server `--base-url` or `ADA_BASE_URL` names. What
it sends is listed [below](#what-ada-chat-sends).

**`ada mcp`.** Answers the app that starts it. With `--http` it listens on
127.0.0.1 and refuses requests from web pages on other hosts; see
[below](#the-mcp-server).

The hosted app at automated-data-analyst.streamlit.app and the optional AI
layer are part of [ADA Lab](https://github.com/saineshnakra/ada-lab), not of
this package.

## What always runs locally

| Step | Where it runs |
|---|---|
| File parsing | Local |
| Cleaning and type inference | Local |
| Schema detection | Local |
| KPIs and evidence | Local |
| Anomaly detection and forecasting | Local |
| Ask ADA parsing and execution | Local |
| Claim checks and folder scans | Local |

A question is turned into a plan by fixed rules and run with pandas; no model
is involved.

## What `ada chat` sends

Each request to the model server carries:

- ADA's instructions (`tools.SYSTEM_PROMPT`) and the three tool definitions
- the paths of the data files under the root, with their row counts and
  column names when there are only a few files
- your questions and the conversation so far
- each tool result, cut to what a model needs (`tools.brief`): the answer, the
  calculation, the file, the rows used, and any assumption, warning or
  question

The file itself is not sent; the model sees the data only through ADA's
answers, which can name values from it, such as the groups an answer
compares. When `ADA_API_KEY` is set, it goes to the same server, for servers
that ask for a key. A server on this computer is reached directly, never
through a proxy the environment names.

## The MCP server

`ada mcp` serves the same three tools to the app that starts it, and that app
passes their results to whatever model it uses. The tools read files under
one root folder (`--root`; by default the folder the server starts in, or
your home folder when it starts at the top of the disk) and refuse a path
outside it.

With `--http`, the server listens on 127.0.0.1, port 8000, unless `--host`
and `--port` say otherwise. A request from a web page on another host is
refused unless `--allow-origin` names that page's origin, so a web page
cannot reach the tools through your browser. With `--host 0.0.0.0`, anything
that can reach that port can read the tables under the root.

## Files ADA writes

ADA writes a file only when a command is told to: `ada scan --template`
writes the answers file it is given, and `ada skill` writes the skill's
files. Nothing else is written to disk.
