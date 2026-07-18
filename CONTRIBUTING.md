# Contributing to ADA

ADA should make business data understandable to someone who has never used a BI tool. Contributions are welcome when they improve accuracy, explainability, usability, privacy, or maintainability.

First time here? Pick something from the [good first issue](https://github.com/saineshnakra/automated-data-analyst/labels/good%20first%20issue) label — each one carries context, file pointers, and acceptance criteria — or say hello in an issue and we will help you scope one.

## Good places to start

- Teach Ask ADA a new question shape (comparisons, shares, date ranges) in `nlq.py` with tests
- Add deterministic business metrics with explicit calculations and tests
- Expand schema detection using realistic, synthetic fixtures
- Improve keyboard navigation, contrast, chart descriptions, or mobile behavior
- Add safe tabular formats without weakening upload validation
- Add adversarial datasets for sparse, messy, or misleading business data
- Improve report exports or contributor documentation

See [ROADMAP.md](ROADMAP.md) for scoped ideas. For a larger change, open a feature request before investing heavily so the product and trust boundaries are clear.

## Development setup

```bash
git clone https://github.com/saineshnakra/automated-data-analyst.git
cd automated-data-analyst
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements-dev.txt
```

Run the application with `streamlit run app.py`.

## Architecture rules

- Put deterministic calculations in `analysis.py`, `business_insights.py`, `nlq.py`, `anomalies.py`, or `forecasting.py`, not in UI callbacks.
- Keep `app.py` focused on orchestration and session state; reusable presentation belongs in `ui.py`.
- Keep model behavior optional and isolated in `ai_insights.py`; model-planned queries must execute through the same local engine as rule-parsed ones.
- Never send raw uploaded rows or cell values to an external model, and never execute model-generated code.
- Treat evidence as observed calculation and recommendations as interpretation.
- Add a test for every bug fix and every new analytical rule.
- Use synthetic data in tests and documentation. Do not commit customer or employer data.

## Pull request checklist

Before opening a pull request:

```bash
ruff check .
python -m unittest discover -s tests -v
python -m compileall -q analysis.py ai_insights.py anomalies.py business_insights.py demo_data.py file_io.py forecasting.py nlq.py pipeline.py ui.py app.py tests
```

In the pull request, explain:

1. The user problem and why it matters
2. The behavior before and after the change
3. The calculation or trust boundary affected
4. How the change was tested
5. Screenshots for visible UI changes

Keep pull requests focused. Avoid unrelated formatting churn or dependency additions without a concrete product need.

## Analytical standard

A useful insight is reproducible from visible evidence. New recommendations must name their supporting calculation, avoid invented causality, and degrade honestly when the data is insufficient. If a heuristic can produce a misleading result, add the edge case to the tests and state the limitation in the interface.

## Conduct and security

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Report vulnerabilities privately according to [SECURITY.md](SECURITY.md).
