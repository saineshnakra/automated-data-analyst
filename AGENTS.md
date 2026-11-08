# Working on ADA

ADA is a Python package (`ada/`) with a command line (`ada`), tool
definitions for language models (`ada/tools.py`), an MCP server
(`ada/mcp.py`) and a chat command (`ada/chat.py`).
[docs/architecture.md](docs/architecture.md) says which module does what.
The research workflow, the modelling mode and the browser app are in
[ADA Lab](https://github.com/saineshnakra/ada-lab); `app.py` here only runs
ADA Lab's app for the hosted demo.

## Setup and checks

```bash
pip install -e ".[dev]"
ruff check .
python -m unittest discover -s tests
```

One test file: `python -m unittest discover -s tests -p "test_claims.py"`.
Before a merge, run `python tools/check.py python3.11 python3.12 python3.13`:
the same checks on each version, plus a build of the wheel that runs
`ada claim`, `ada ask`, `ada scan` and `ada mcp` from it. GitHub Actions does
not run on this account, so CI does not run on pushes or pull requests.

## Conventions

- Calculations live in the engine modules, never in the command line, the
  tools or the MCP server; those only wrap the engine.
- Every number comes from pandas, SciPy, statsmodels or SymPy and is shown with its
  calculation. When something can't be computed honestly, refuse with a
  reason instead of guessing.
- Results are deterministic: the same data and question give the same
  answer. Don't put dates, temporary paths or random state in outputs.
- Code is formatted by hand: ruff check only, line length 110. Don't run
  `ruff format`.
- Tests use `unittest`. A new rule needs a test, including how it behaves on
  thin data.
- Docs, captions and messages are plain English in the present tense, with
  no marketing words.
- Commit messages are one sentence that says what changed, such as "Skip
  empty numeric columns in the value checks instead of failing on them".
