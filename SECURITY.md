# Security policy

## Data handling

ADA processes uploaded CSV and Excel files in the active Streamlit session. The deterministic dashboard does not require an API key, execute generated code, or intentionally persist uploaded datasets.

If the optional strategy layer is enabled, ADA sends only computed schema roles, summaries, evidence, deterministic recommendations, and context typed by the user. Raw uploaded rows are not included in the model prompt. Model responses use a strict typed schema, request storage is disabled, and the request includes a hashed anonymous session identifier. Review `build_ai_payload` in `ai_insights.py` when changing this boundary.

Hosted infrastructure still processes network traffic and application memory. Do not upload data you are not authorized to place on the selected deployment. The no-raw-row boundary reduces disclosure; it does not turn a public hosted app into an approved environment for regulated data.

## Secrets

An API credential is optional. A visitor-provided key remains in the active Streamlit session and is sent only to the model API. For a trusted deployment, store `OPENAI_API_KEY` in the process environment or Streamlit's secret manager. Do not place an owner-funded key on a public app without authentication, rate limits, and spending controls.

Never commit `.env`, `.streamlit/secrets.toml`, service-account files, access tokens, sample production data, or screenshots containing private information. Common local secret files are ignored by Git.

If a credential is exposed, revoke it at the provider immediately, remove it from Git history, and rotate any downstream credential derived from it.

## Supported versions

Security fixes are applied to the latest commit on `main`. Older snapshots are not separately maintained.

## Reporting a vulnerability

Report security concerns privately through the contact information at [sainesh.com](https://sainesh.com/) rather than opening a public issue with sensitive details. Include the affected path, impact, reproduction steps, and a safe proof of concept when possible.
