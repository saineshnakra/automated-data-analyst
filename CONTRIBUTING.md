# Contributing to ADA

ADA should make business data understandable to someone who has never used a BI
tool. Contributions are welcome when they improve accuracy, explainability,
usability, privacy, or maintainability.

New here? Pick a [good first issue](https://github.com/saineshnakra/automated-data-analyst/labels/good%20first%20issue)
— each one has context, file pointers, and acceptance criteria. Or open an issue
and we will help you scope one.

## Before you start

| Read | For |
|---|---|
| [docs/concepts.md](docs/concepts.md) | The vocabulary — measure, segment, period, evidence, plan |
| [docs/how-it-works.md](docs/how-it-works.md) | What happens between upload and dashboard |
| [docs/architecture.md](docs/architecture.md) | Which file owns what, and the rules that keep it that way |
| [docs/development.md](docs/development.md) | Setup, tests, linting, CI, conventions |

Then the [reference page](docs/README.md#reference) for the module you are
touching. Each one ends with a "Changing this" section.

## Good places to start

- Teach Ask ADA a new question shape — comparisons, shares, date ranges
  ([reference](docs/reference/ask-ada.md))
- Add a deterministic metric with an explicit calculation
  ([reference](docs/reference/evidence.md))
- Expand schema detection with realistic synthetic fixtures
  ([reference](docs/reference/schema-detection.md))
- Improve keyboard navigation, contrast, chart descriptions, or mobile behavior
- Add safe tabular formats without weakening upload validation
  ([reference](docs/reference/reading-files.md))
- Add adversarial datasets for sparse, messy, or misleading business data

See [ROADMAP.md](ROADMAP.md) for scoped ideas. For a larger change, open a
feature request first so the product and trust boundaries are agreed before you
invest heavily.

## The rules that are not negotiable

These protect the thing that makes ADA worth using. Full reasoning in
[docs/architecture.md](docs/architecture.md#rules-that-hold-the-design-together).

1. Calculations go in the engine modules, never in a UI callback.
2. `aggregation.py` is the only place that decides what a period is.
3. Formatting changes how a value looks, never what it is.
4. Evidence is calculated; recommendations are interpretation. Keep them apart,
   and never claim cause.
5. Raw rows never reach a model, and no cell value does either, with one
   documented exception: an evidence sentence names the segment it describes,
   so a segment label can travel inside the strategic read
   ([docs/privacy.md](docs/privacy.md)). Model-generated code is never
   executed.
6. Every analytical rule gets a test, including how it degrades on thin data.
7. The product works with no API key.

## Before opening a pull request

```bash
ruff check .
python -m unittest discover -s tests -v
```

In the pull request body, explain:

1. The user problem and why it matters
2. The behavior before and after
3. The calculation or trust boundary affected
4. How you tested it
5. Screenshots for visible UI changes

Keep pull requests focused. Avoid unrelated formatting churn, and do not add a
dependency without a concrete product need.

## Analytical standard

A useful insight is reproducible from visible evidence. New recommendations must
name their supporting calculation, avoid invented causality, and degrade
honestly when the data is insufficient. If a heuristic can produce a misleading
result, add the edge case to the tests and state the limitation in the
interface.

Use synthetic data in tests and documentation. Never commit customer or employer
data.

## Conduct and security

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Report
vulnerabilities privately per [SECURITY.md](SECURITY.md).
