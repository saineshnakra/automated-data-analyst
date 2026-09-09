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

## Size: about 100 lines of code

**A pull request should change at most ~100 lines of executable code.** Prose,
tests and fixtures do not count against it — a 40-line fix with 300 lines of
tests and a long explanation is exactly right, and a 400-line refactor with no
tests is exactly wrong.

This is not tidiness. Review quality falls off a cliff somewhere around a
hundred lines: past that, a reviewer reads for *plausibility* rather than for
*bugs*, and approves things nobody actually checked. That failure has already
happened in this repository — a large change here shipped with a green suite of
385 tests and still carried eight defects, including a forecast that printed
`nan` and a row cap that could reduce a 250,000-row upload to nothing. Every one
of them was obvious in a fifty-line diff and invisible in a four-thousand-line
one.

So if your change is bigger than that, split it. In order of preference:

1. **By behaviour.** One user-visible change per pull request, with the tests
   that prove it. "Rates are averaged, not summed" is one; "a rate moves in
   percentage points" is another, even though they touch the same file.
2. **Groundwork first.** A new module, a shared helper or a renamed function
   with no behaviour change is its own pull request, and it is a *small* one to
   review because nothing should move. The change that uses it comes next.
3. **As a stack.** Open them in order, each based on the last, and say so in the
   body: `Stacked on #123 — review that first`. Merge from the bottom up.

Two things that do **not** count as splitting: shipping a large change as
several commits in one pull request (a reviewer still faces the whole diff), and
holding tests back for a follow-up (a change and its tests are one unit).

Generated or vendored files, lockfiles, and mechanical renames applied by a tool
are exempt — say which in the body so a reviewer can skip them with confidence.

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
