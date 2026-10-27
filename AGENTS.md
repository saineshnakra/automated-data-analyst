# Working on ADA

ADA is a Python package (`ada/`) with a command line (`ada`), a research
workflow (`ada/research/`: `ada init`, `ada build`, `ada verify`) and a
Streamlit app (`app.py`, `ada/ui.py`, `ada/home.py`).
[docs/architecture.md](docs/architecture.md) says which module does what.

## Setup and checks

```bash
pip install -e ".[app,ai,dev,model]"
ruff check .
python -m unittest discover -s tests
```

One test file: `python -m unittest discover -s tests -p "test_research_build.py"`.
CI runs the same checks on Python 3.11, 3.12 and 3.13, plus a job that builds
the wheel and runs `ada init`, `ada build` and `ada verify` from it.

## Conventions

- Calculations live in the engine modules and `ada/research/`, never in the
  app; the app only draws results.
- Every number comes from pandas, SciPy or statsmodels and is shown with its
  calculation. When something can't be computed honestly, refuse with a
  reason instead of guessing.
- Builds are byte-stable: the same study.yml and data give the same files.
  Don't put dates, temporary paths or random state in outputs.
- Code is formatted by hand: ruff check only, line length 110. Don't run
  `ruff format`.
- Tests use `unittest`. A new rule needs a test, including how it behaves on
  thin data.
- Docs, captions and messages are plain English in the present tense, with
  no marketing words.
- Commit messages are one sentence that says what changed, such as "Skip
  empty numeric columns in the value checks instead of failing on them".
