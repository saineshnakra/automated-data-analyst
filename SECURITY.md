# Security policy

## Data handling

ADA processes uploaded CSV and Excel files in the server process running the
Streamlit app, or in your own process when used as a library. The analysis
needs no API key, executes no generated code, and does not write uploaded data
to disk. Parsed uploads are held in an in-memory cache (at most eight files,
up to an hour each).

The optional AI features send only the column schema, computed summaries and
evidence, the deterministic recommendations, and text the user typed (business
context, or a question the rule parser could not read). Raw rows are not
included. An evidence sentence can name a segment value, such as a customer or
product. Responses are parsed into strict typed models, prompt storage is
disabled, and each request carries a hashed anonymous session identifier. A
query plan from the model runs locally and only after the user approves it.
Review `build_ai_payload` and `build_planner_payload` in `ada/ai_insights.py`
when changing this boundary. Details are in [docs/privacy.md](docs/privacy.md).

The cleaned-data CSV download escapes text cells and headers that start with
`=`, `+`, `-`, `@`, a tab or a carriage return, so a spreadsheet does not
evaluate them as formulas (`safe_csv` in `ada/file_io.py`).

A hosted deployment still processes network traffic and holds data in memory.
Do not upload data you are not authorized to place on that deployment. Keeping
raw rows away from the model reduces disclosure; it does not make a public
hosted app suitable for regulated data.

## Secrets

An API key is optional. A key entered in the sidebar stays in the active
Streamlit session and is sent only to the model API. For a trusted deployment,
set `OPENAI_API_KEY` in the process environment or in Streamlit's secrets. Do
not put an owner-funded key on a public app without authentication, rate
limits and spending controls.

Never commit `.env`, `.streamlit/secrets.toml`, service-account files, access
tokens, production data, or screenshots containing private information.
Common local secret files are listed in `.gitignore`.

If a credential is exposed, revoke it at the provider, remove it from Git
history, and rotate any credential derived from it.

## Supported versions

Security fixes are applied to the latest commit on `main`. Older versions are
not maintained separately.

## Reporting a vulnerability

Report security issues privately through the contact details at
[sainesh.com](https://sainesh.com/), not in a public issue. Include the
affected file or URL, the impact, steps to reproduce, and a safe proof of
concept where possible.
