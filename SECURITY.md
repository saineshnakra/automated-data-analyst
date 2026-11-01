# Security policy

## Data handling

ADA runs in your own process, as the `ada` command or as a Python library. It
reads the CSV and Excel files it is given, needs no API key, executes no
generated code, and writes only the files a command is told to write (such as
`ada scan --template` and `ada skill`). It makes no network call, with two
exceptions:

- `ada chat` sends your questions, the paths and column names of the data
  files, and ADA's results to the model server it is pointed at: Ollama on
  this computer by default, or the server `--base-url` or `ADA_BASE_URL`
  names. A result can name values from the data, such as group names.
- `ada mcp --http` listens on 127.0.0.1 unless `--host` says otherwise, and
  refuses a request from a web page on another host unless `--allow-origin`
  names that origin. With `--host 0.0.0.0`, anything that can reach the port
  can read the tables under the root folder.

The tools (`ada.tools.run` with a `root`, `ada mcp` and `ada chat`) read only
files under their root folder, and refuse a path outside it.

`safe_csv` in `ada/file_io.py`, which turns a table into CSV text, escapes
text cells and headers that start with `=`, `+`, `-`, `@`, a tab or a
carriage return, so a spreadsheet does not evaluate them as formulas.

The hosted app at automated-data-analyst.streamlit.app, which uploads data to
a server, and the optional AI layer, which calls a hosted model, are part of
[ADA Lab](https://github.com/saineshnakra/ada-lab). Details for ADA are in
[docs/privacy.md](docs/privacy.md).

## Secrets

ADA needs no API key. When `ADA_API_KEY` is set, `ada chat` sends it to the
model server it talks to, and nowhere else.

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
