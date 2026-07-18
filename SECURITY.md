# Security

## Data handling

Automated Data Analyst processes uploaded CSV and Excel files in the active Streamlit session. The application does not require an OpenAI key, call an LLM API, execute generated code, or intentionally persist uploaded datasets.

As with any hosted application, the infrastructure provider still processes network traffic and application memory. Do not upload data that you are not authorized to place on the selected Streamlit deployment.

## Secrets

No API credentials are required. Never commit `.env` files, Streamlit secrets, service-account files, or access tokens. The repository's ignore rules cover common local secret files.

## Reporting a vulnerability

Please report security concerns privately through the contact information at [sainesh.com](https://sainesh.com/) rather than opening a public issue containing sensitive details.
