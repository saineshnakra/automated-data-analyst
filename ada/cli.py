"""The ``ada`` command.

    ada analyze orders.csv
    ada schema orders.csv
    ada ask orders.csv "which category made the most revenue"
    ada check metrics.csv --measure signups --frequency W
    ada samples
    ada app

``check`` is the one built for cron. Its exit code is the answer, so a shell
can branch without parsing anything:

    0  looked, nothing outside the band
    1  looked, something was flagged
    3  could not look -- no timeline, or too little history
    2  the input could not be used at all

Three and one are different on purpose. A job that cannot tell "quiet" from
"blind" will eventually report quiet for a week because its feed broke.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .api import AdaError, Refusal, analyze, sample, samples

EXIT_QUIET = 0
EXIT_FLAGGED = 1
EXIT_ERROR = 2
EXIT_COULD_NOT_LOOK = 3


def _load(args: argparse.Namespace):
    source = sample() if args.file == "-" else args.file
    return analyze(
        source,
        date=getattr(args, "date", None),
        measure=getattr(args, "measure", None),
        dimension=getattr(args, "dimension", None),
        frequency=getattr(args, "frequency", None),
    )


def _analyze(args: argparse.Namespace) -> int:
    report = _load(args)
    print(json.dumps(report.to_dict(), indent=2) if args.json else report)
    return EXIT_QUIET


def _schema(args: argparse.Namespace) -> int:
    """What ADA decided, so you know which names to pass back to it."""
    report = _load(args)
    roles = report.roles
    print(f"{report.source_name} — {report.rows_analyzed:,} rows")
    for label, value in (
        ("date", roles.date), ("measure", roles.measure),
        ("segment", roles.dimension), ("identifier", roles.identifier),
    ):
        print(f"  {label:11} {value or '(none detected)'}")
    if report.cleaning.notes:
        print("\ncleaning:")
        for note in report.cleaning.notes:
            print(f"  - {note}")
    return EXIT_QUIET


def _ask(args: argparse.Namespace) -> int:
    answer = _load(args).ask(args.question)
    if isinstance(answer, Refusal):
        print(f"refused ({answer.code}): {answer.message}", file=sys.stderr)
        return EXIT_COULD_NOT_LOOK
    print(answer.answer)
    print(f"  calc: {answer.calculation}")
    return EXIT_QUIET


def _check(args: argparse.Namespace) -> int:
    report = _load(args)
    found = report.anomalies
    if isinstance(found, Refusal):
        print(f"could not look ({found.code}): {found.message}", file=sys.stderr)
        return EXIT_COULD_NOT_LOOK
    if not found:
        print(f"{report.source_name}: nothing outside the expected band.")
        return EXIT_QUIET
    for item in found:
        key = f"anomaly:{item.period:%Y-%m-%d}"
        print(f"{key}  {item.direction} to {item.value:,.2f}, expected about {item.expected:,.2f}")
    return EXIT_FLAGGED


def _samples(_: argparse.Namespace) -> int:
    for name, path in sorted(samples().items()):
        print(f"{name:22} {path}")
    return EXIT_QUIET


def _app(_: argparse.Namespace) -> int:
    """Launch the Streamlit app, which lives behind the [app] extra."""
    try:
        from streamlit.web import cli as streamlit_cli  # noqa: PLC0415
    except ImportError:
        print(
            "The app needs Streamlit. Install it with:\n"
            "    pip install 'automated-data-analyst[app]'",
            file=sys.stderr,
        )
        return EXIT_ERROR
    from pathlib import Path  # noqa: PLC0415

    entry = Path(__file__).resolve().parent.parent / "app.py"
    if not entry.is_file():
        print(f"Could not find the app at {entry}.", file=sys.stderr)
        return EXIT_ERROR
    sys.argv = ["streamlit", "run", str(entry)]
    streamlit_cli.main()
    return EXIT_QUIET


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ada",
        description="Exploratory analysis for any table, with the arithmetic under every number.",
    )
    parser.add_argument("--version", action="version", version=f"ada {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    def with_source(name: str, help_text: str):
        sub = subcommands.add_parser(name, help=help_text)
        sub.add_argument("file", help="a CSV or Excel file, or - for a bundled sample")
        sub.add_argument("--date", help="override the detected date column")
        sub.add_argument("--measure", help="override the detected measure column")
        sub.add_argument("--dimension", help="override the detected segment column")
        sub.add_argument("--frequency", choices=list("DWMQY"), help="pin the period grain")
        return sub

    with_source("analyze", "read a table and print what is in it").add_argument(
        "--json", action="store_true", help="machine-readable output"
    )
    with_source("schema", "print the columns ADA detected and what it cleaned")
    with_source("ask", "ask a question in plain English").add_argument("question")
    with_source("check", "flag periods outside the expected band; for cron")

    subcommands.add_parser("samples", help="list the bundled datasets")
    subcommands.add_parser("app", help="launch the browser app (needs the app extra)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = {
        "analyze": _analyze, "schema": _schema, "ask": _ask,
        "check": _check, "samples": _samples, "app": _app,
    }[args.command]
    try:
        return handler(args)
    except AdaError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
