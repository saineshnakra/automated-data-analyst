"""The ``ada`` command.

Compute an expression or solve equations exactly, with every step:

    ada compute "0.15 * (1200 - 300)"        # 135; fractions stay exact, so 1/3 + 1/6 is 1/2
    ada compute "comb(52, 5)" --json
    ada compute "x + y = 10; x - y = 2"      # x = 6, y = 4, checked by putting them back
    ada compute "3(x + 2) - x"               # 2x + 6

``ada compute`` exits 0 with the value, or 3 when the expression cannot be
computed honestly (division by zero, an expression that reads two ways, a name
it does not know), with the reason.

Give a language model ADA's tools:

    ada chat penguins.csv              # chat with a local model that asks ADA for every number
    ada mcp                            # serve the tools over MCP, for LM Studio, Jan, Claude Desktop and more
    ada tools --style anthropic        # print the tool definitions for function calling
    ada skill                          # install the skill that tells Claude Code and Codex how to use ADA

For a first look at any table:

    ada analyze measurements.csv
    ada schema measurements.csv
    ada ask orders.csv "which category made the most revenue"       # --json for programs
    ada claim flights.csv "The average dep_delay was 12.64 minutes."   # check a stated number
    ada scan data/                     # every table in a folder, how they join, what to confirm
    ada check metrics.csv --measure signups --frequency W
    ada check rainfall.csv --measure "Rainfall (mm)" --aggregation sum
    ada samples

``ada init``, ``ada build``, ``ada verify``, ``ada model``, ``ada app`` and the other
commands of the research workflow, the modelling mode and the browser app are in ADA
Lab (``pip install ada-lab``); ``ada`` hands them to it when it is installed.

``ada claim`` checks one number someone states about a table and says verified, or
unverified with the reason (see ``ada.claims``). It is not ``ada check``, which
watches a measure for unusual periods. It reports through its exit code too:

    0  verified
    1  unverified: the data give a different number, or a check on the analysis failed
    2  the data could not be read
    3  unverified for another reason: missing context, ambiguous wording, unsupported

An answer lists the choices it rests on (``decides:``, or ``decisions`` in the
JSON), each with ADA's default and the other options; ``--decide KEY=OPTION``
picks another, such as ``--decide aggregate_rows=keep`` to count a World row
with the countries.

``ada scan PATH --template answers.yml`` writes the questions to ask before an
analysis (what it should find out, what a row is, units, links, decisions) with
ADA's proposed answers; ``ada ask`` and ``ada claim`` take the edited file with
``--answers answers.yml``. ``ada ask`` also takes a folder: the question is
answered from the file it fits, or from two files joined along a link.

``ada scan FOLDER`` profiles each table in a folder, proposes the keys that join
them (with the rows each join would leave unmatched) and lists what a person
should confirm, grouped by topic. ``--template FILE`` writes those questions with
ADA's proposed answers to a YAML file; edit it and pass it back with ``--answers
FILE``, and confirmed answers are not asked again. ``ada claim`` also takes a
folder and checks the claim against the file it fits.

``ada check`` is meant for scheduled jobs and reports through its exit code:

    0  checked, nothing flagged
    1  checked, at least one period flagged
    2  the input could not be read
    3  could not check: no date column or too few periods

3 is kept separate from 0 so that a broken feed doesn't look like a quiet one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .api import AdaError, Refusal, analyze, sample, samples
from .claims import check_claim
from .decisions import OPTIONS, UnknownDecision, parse_choices
from .folder import load_answers, scan
from .joins import ask_folder, folder_choices
from .metrics import AGGREGATION_CHOICES, metric_for
from .nlq import ANSWER_SCHEMA_VERSION

EXIT_QUIET = 0
EXIT_FLAGGED = 1
EXIT_ERROR = 2
EXIT_COULD_NOT_LOOK = 3

#: Commands that moved to ADA Lab (the ada-lab package); ``ada`` hands them to it when it is installed.
MOVED_COMMANDS = ("init", "build", "verify", "preview", "share", "demo", "research", "model", "auto", "app")

DECIDE_HELP = (
    "pick another option for a choice the answer rests on, such as aggregate_rows=keep; "
    "repeat for more (" + "; ".join(f"{key}: {', '.join(options)}" for key, options in OPTIONS.items()) + ")"
)
ANSWERS_HELP = "the answers file `ada scan --template` writes; its decisions are used, and --decide wins"


def _load(args: argparse.Namespace):
    source = sample() if args.file == "-" else args.file
    return analyze(
        source,
        sheet=getattr(args, "sheet", None),
        date=getattr(args, "date", None),
        measure=getattr(args, "measure", None),
        dimension=getattr(args, "dimension", None),
        frequency=getattr(args, "frequency", None),
        aggregation=getattr(args, "aggregation", None),
    )


def _analyze(args: argparse.Namespace) -> int:
    report = _load(args)
    print(json.dumps(report.to_dict(), indent=2) if args.json else report)
    return EXIT_QUIET


def _schema(args: argparse.Namespace) -> int:
    """Print the detected column roles and any cleaning notes."""
    report = _load(args)
    roles = report.roles
    measure = roles.measure and f"{roles.measure} ({metric_for(roles.measure, roles).aggregation})"
    print(f"{report.source_name} — {report.rows_analyzed:,} rows")
    for label, value in (
        ("date", roles.date), ("measure", measure),
        ("segment", roles.dimension), ("identifier", roles.identifier),
    ):
        print(f"  {label:11} {value or '(none detected)'}")
    if report.cleaning.notes:
        print("\ncleaning:")
        for note in report.cleaning.notes:
            print(f"  - {note}")
    return EXIT_QUIET


def _choices(args: argparse.Namespace) -> dict[str, str] | None:
    """The --decide choices, checked; None after printing why one is wrong or the answers file is missing."""
    if getattr(args, "answers", None) and not Path(args.answers).is_file():
        print(f"error: the answers file {args.answers} does not exist; write it with ada scan --template.",
              file=sys.stderr)
        return None
    try:
        return parse_choices(args.decide or [])
    except UnknownDecision as error:
        print(f"error: {error}", file=sys.stderr)
        return None


def _ask(args: argparse.Namespace) -> int:
    if (choices := _choices(args)) is None:
        return EXIT_ERROR
    try:
        if Path(args.file).is_dir():
            answer = ask_folder(args.file, args.question, decisions=choices, answers=args.answers)
        else:
            given = load_answers(args.answers)
            if given:
                choices = folder_choices(given, Path(args.file).name, choices)
            answer = _load(args).ask(args.question, decisions=choices)
    except UnknownDecision as error:
        print(f"error: {error} (in {args.answers})", file=sys.stderr)
        return EXIT_ERROR
    if isinstance(answer, Refusal):
        if args.json:
            print(json.dumps({
                "schema_version": ANSWER_SCHEMA_VERSION, "status": "refused", "question": args.question,
                "code": answer.code, "message": answer.message,
            }, indent=2))
        else:
            print(f"refused ({answer.code}): {answer.message}", file=sys.stderr)
        return EXIT_COULD_NOT_LOOK
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2))
        return EXIT_QUIET
    print(answer.answer)
    print(f"  calc: {answer.calculation}")
    for decision in answer.decisions:
        print(f"  decides: {decision.describe()}")
    return EXIT_QUIET


def _claim(args: argparse.Namespace) -> int:
    if args.claim is None and (args.question is None or args.value is None):
        print("error: give the claim as a sentence, or both --question and --value.", file=sys.stderr)
        return EXIT_ERROR
    if (choices := _choices(args)) is None:
        return EXIT_ERROR
    try:
        check = check_claim(
            args.file, args.claim, question=args.question, value=args.value, sheet=args.sheet,
            decisions=choices, answers=args.answers,
        )
    except UnknownDecision as error:
        print(f"error: {error} (in {args.answers})", file=sys.stderr)
        return EXIT_ERROR
    print(json.dumps(check.to_dict(), indent=2) if args.json else check)
    if check.verified:
        return EXIT_QUIET
    if check.reason == "failed_check":
        return EXIT_FLAGGED
    return EXIT_ERROR if check.reason == "inaccessible_data" else EXIT_COULD_NOT_LOOK


def _scan(args: argparse.Namespace) -> int:
    try:
        found = scan(args.folder, answers=args.answers)
    except NotADirectoryError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR
    if args.template:
        Path(args.template).write_text(found.answers_template())
    print(json.dumps(found.to_dict(), indent=2) if args.json else found)
    if args.template and not args.json:
        print(f"\nWrote the questions to {args.template}. Edit the answers, then run "
              f"`ada scan {args.folder} --answers {args.template}`.")
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
        print(f"{key}  {item.value:,.2f}, {item.direction} the expected {item.expected:,.2f}")
    return EXIT_FLAGGED


#: Where each assistant looks for skills, in the project folder or the home folder: Claude Code reads
#: .claude/skills, and Codex and other tools that follow the Agent Skills format read .agents/skills.
SKILL_FOLDERS = {"claude": Path(".claude", "skills"), "codex": Path(".agents", "skills")}
#: The skill's files, by their path in its folder. agents/openai.yaml is how Codex names and shows it.
SKILL_FILES = ("SKILL.md", "agents/openai.yaml")


def _skill(args: argparse.Namespace) -> int:
    """Install or check ADA's agent skill where Claude Code and Codex look for skills."""
    from importlib.resources import files  # noqa: PLC0415

    if args.check and (args.print or args.force):
        print("error: --check cannot be used with --print or --force.", file=sys.stderr)
        return EXIT_ERROR
    source = files("ada").joinpath("skills", "ada")
    contents = {name: source.joinpath(*name.split("/")).read_text(encoding="utf-8") for name in SKILL_FILES}
    if args.print:
        print(contents["SKILL.md"], end="")
        return EXIT_QUIET
    if args.dir:
        folders = {"": Path(args.dir)}
    else:
        base = Path.home() if args.user else Path()
        chosen = SKILL_FOLDERS if args.target == "all" else {args.target: SKILL_FOLDERS[args.target]}
        folders = {assistant: base / folder for assistant, folder in chosen.items()}

    def differs(path: Path, text: str) -> bool:
        return not path.is_file() or path.read_text(encoding="utf-8") != text

    targets = {assistant: folder / "ada" for assistant, folder in folders.items()}
    if args.check:
        status = EXIT_QUIET
        for target in targets.values():
            for name, text in contents.items():
                path = target / name
                try:
                    installed = path.read_text(encoding="utf-8")
                except FileNotFoundError:
                    print(f"{path} is missing (skill file not installed).")
                except UnicodeDecodeError:
                    print(f"{path} differs from this version's skill.")
                except OSError as error:
                    print(f"error: cannot read {path}: {error}", file=sys.stderr)
                    status = EXIT_ERROR
                    continue
                else:
                    if installed == text:
                        print(f"{path} is already this version's skill.")
                        continue
                    print(f"{path} differs from this version's skill.")
                if status != EXIT_ERROR:
                    status = EXIT_FLAGGED
        return status
    edited = [
        target / name
        for target in targets.values()
        for name, text in contents.items()
        if (target / name).exists() and differs(target / name, text)
    ]
    if edited and not args.force:
        listed = ", ".join(str(path) for path in edited)
        print(f"error: {listed} already exists and differs from this version's skill. "
              "Pass --force to replace it.", file=sys.stderr)
        return EXIT_ERROR
    names = {"claude": "Claude Code", "codex": "Codex and other assistants that read .agents/skills", "": ""}
    for assistant, target in targets.items():
        written = [name for name, text in contents.items() if differs(target / name, text)]
        for name in written:
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            (target / name).write_text(contents[name], encoding="utf-8")
        reader = f" for {names[assistant]}" if names[assistant] else ""
        print(f"Wrote {target}{reader}." if written else f"{target} is already this version's skill.")
    return EXIT_QUIET


def _tools(args: argparse.Namespace) -> int:
    from .tools import definitions  # noqa: PLC0415

    print(json.dumps(definitions(args.style), indent=2))
    return EXIT_QUIET


def _chat(args: argparse.Namespace) -> int:
    from .chat import Chat, ChatError, shown  # noqa: PLC0415

    try:
        session = Chat(args.data, model=args.model, base_url=args.base_url)
    except ChatError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR

    def show(call) -> None:
        if not args.json:
            print(shown(call), file=sys.stderr if args.question else sys.stdout, flush=True)

    def answer(question: str) -> int:
        try:
            reply = session.ask(question, on_call=show)
        except ChatError as error:
            print(f"error: {error}", file=sys.stderr)
            return EXIT_ERROR
        print(json.dumps(reply.to_dict(), indent=2, ensure_ascii=False) if args.json else reply.answer)
        return EXIT_QUIET

    if args.question:
        return answer(" ".join(args.question))
    print(f"ADA chat: {session.model} at {session.base_url}, reading {args.data}. Ask about the data; "
          "Ctrl-D or 'exit' leaves.")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return EXIT_QUIET
        if question.lower() in ("exit", "quit"):
            return EXIT_QUIET
        if question and answer(question) == EXIT_ERROR:
            return EXIT_ERROR


def _mcp(args: argparse.Namespace) -> int:
    from .mcp import http_server, serve_stdio  # noqa: PLC0415

    root = Path(args.root).expanduser() if args.root else None
    if root is not None and not root.is_dir():
        print(f"error: {args.root} is not a folder", file=sys.stderr)
        return EXIT_ERROR
    if not args.http:
        for stream in (sys.stdin, sys.stdout):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
        serve_stdio(root)
        return EXIT_QUIET
    server = http_server(root, host=args.host, port=args.port, allowed_origins=tuple(args.allow_origin or ()))
    from .mcp import Server  # noqa: PLC0415

    print(f"ADA's MCP server is at http://{args.host}:{server.server_address[1]}/mcp, reading files under "
          f"{Server(root).root}. Ctrl-C stops it.", file=sys.stderr, flush=True)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"It listens on {args.host}: anything that can reach this computer on port "
              f"{server.server_address[1]} can read the tables under that folder.", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return EXIT_QUIET


def _compute(args: argparse.Namespace) -> int:
    from .algebra import work_out  # noqa: PLC0415

    result = work_out(" ".join(args.expression), digits=args.digits)
    if isinstance(result, Refusal):
        if args.json:
            print(json.dumps({"schema_version": 1, "status": "refused", "code": result.code,
                              "message": result.message}, indent=2))
        else:
            print(f"refused ({result.code}): {result.message}", file=sys.stderr)
        return EXIT_COULD_NOT_LOOK
    print(json.dumps(result.to_dict(), indent=2) if args.json else result)
    return EXIT_QUIET


def _compute_arguments(argv: list[str]) -> list[str]:
    """``ada compute -2^2``: everything but compute's own options is the expression, even when it starts with
    a minus sign that would read as an option."""
    options, expression, rest = [], [], iter(argv[1:])
    for item in rest:
        if item in ("--json", "-h", "--help"):
            options.append(item)
        elif item == "--digits":
            options += [item, next(rest, "")]
        elif item.startswith("--digits="):
            options.append(item)
        elif item != "--":
            expression.append(item)
    return ["compute", *options, "--", *expression] if expression else ["compute", *options]


def _samples(_: argparse.Namespace) -> int:
    for name, path in sorted(samples().items()):
        print(f"{name:22} {path}")
    return EXIT_QUIET


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ada",
        description="Compute and check numbers from CSV and Excel data, for people and language models.",
        epilog="The research workflow (init, build, verify, preview, share, demo), the modelling mode "
               "(model, auto) and the browser app (app) are in ADA Lab: pip install ada-lab.",
    )
    parser.add_argument("--version", action="version", version=f"ada {__version__}")
    subcommands = parser.add_subparsers(dest="command", metavar="COMMAND")

    def with_source(name: str, help_text: str):
        sub = subcommands.add_parser(name, help=help_text)
        sub.add_argument("file", help="a CSV, TSV or Excel file, or - for a bundled sample")
        sub.add_argument("--sheet", help="the worksheet to read from a workbook (default: the first)")
        sub.add_argument("--date", help="override the detected date column")
        sub.add_argument("--measure", help="override the detected measure column")
        sub.add_argument("--dimension", help="override the detected segment column")
        sub.add_argument("--frequency", choices=["s", "min", "h", *"DWMQY"], help="pin the period grain")
        sub.add_argument(
            "--aggregation", choices=AGGREGATION_CHOICES,
            help="combine the measure's rows as a total (sum) or an average (mean)",
        )
        return sub

    with_source("analyze", "read a table and print what is in it").add_argument(
        "--json", action="store_true", help="machine-readable output"
    )
    with_source("schema", "print the columns ADA detected and what it cleaned")
    asked = with_source("ask", "ask a question in plain English, of a file or a folder")
    asked.add_argument("question")
    asked.add_argument("--json", action="store_true", help="machine-readable answer with its evidence")
    asked.add_argument("--decide", action="append", metavar="KEY=OPTION", help=DECIDE_HELP)
    asked.add_argument("--answers", help=ANSWERS_HELP)
    claimed = subcommands.add_parser(
        "claim", help="check a number stated about a table: verified, or unverified with the reason"
    )
    claimed.add_argument("file", help="the CSV, TSV or Excel file the claim is about, or a folder of them")
    claimed.add_argument("claim", nargs="?", help='the claim, such as "The average delay was 12.6 minutes."')
    claimed.add_argument("--question", help="the calculation, when it is known: \"average dep_delay\"")
    claimed.add_argument("--value", help="the number claimed, with --question; its decimals set precision")
    claimed.add_argument("--sheet", help="the worksheet to read from a workbook (default: the first)")
    claimed.add_argument("--json", action="store_true", help="machine-readable result")
    claimed.add_argument("--decide", action="append", metavar="KEY=OPTION", help=DECIDE_HELP)
    claimed.add_argument("--answers", help=ANSWERS_HELP)
    scanned = subcommands.add_parser(
        "scan", help="profile every table in a folder, propose how they join, and list what to confirm"
    )
    scanned.add_argument("folder", help="the folder of CSV, TSV or Excel files, or one such file")
    scanned.add_argument("--answers", help="a YAML file of answers given before (see --template)")
    scanned.add_argument("--template", help="write the questions and proposed answers to this YAML file")
    scanned.add_argument("--json", action="store_true", help="machine-readable result")
    with_source("check", "flag periods outside the expected band; for cron")

    skill = subcommands.add_parser(
        "skill", help="install the skill that tells Claude Code and Codex how to use ADA"
    )
    skill.add_argument("--for", dest="target", choices=["all", "claude", "codex"], default="all",
                       help="claude (.claude/skills), codex (.agents/skills) or all of them (default)")
    where = skill.add_mutually_exclusive_group()
    where.add_argument("--user", action="store_true",
                       help="write to the skills folders in your home folder, for every project")
    where.add_argument("--dir", help="another assistant's skills folder; writes DIR/ada/")
    where.add_argument("--print", action="store_true", help="print SKILL.md instead of writing it")
    skill.add_argument("--check", action="store_true",
                       help="check without writing; exit 1 if any skill file is missing or differs")
    skill.add_argument("--force", action="store_true", help="replace skill files that differ")

    tooled = subcommands.add_parser(
        "tools", help="print ADA's tool definitions for function calling, so any language model can use ADA"
    )
    tooled.add_argument("--style", choices=["openai", "anthropic", "mcp"], default="openai",
                        help="openai (also Ollama, vLLM, llama.cpp and LM Studio), anthropic or mcp "
                             "(default: openai)")

    chatted = subcommands.add_parser(
        "chat", help="chat with a local model (Ollama, LM Studio, llama.cpp) that uses ADA for every number",
        description="Chat with a model that calls ADA's tools: ADA computes every number, the model answers "
                    "with them. Works with any OpenAI-compatible server; the default is Ollama on this "
                    "computer. The model needs tool calling, such as qwen3, llama3.1 or mistral.",
    )
    chatted.add_argument("data", nargs="?", default=".",
                         help="a data file or a folder of them (default: here)")
    chatted.add_argument("question", nargs="*", help="ask one question and stop, instead of chatting")
    chatted.add_argument("--model", help="the model (default: ADA_MODEL, else the server's only model, else "
                                         "qwen3:1.7b)")
    chatted.add_argument("--base-url", help="the server's OpenAI-compatible API (default: ADA_BASE_URL, else "
                                            "Ollama at http://localhost:11434/v1)")
    chatted.add_argument("--json", action="store_true", help="print the answer and every tool call as JSON")

    served = subcommands.add_parser(
        "mcp", help="run ADA as an MCP server, for LM Studio, Open WebUI, Jan, Claude Desktop, Cursor and "
                    "others",
        description="Serve ADA's tools over the Model Context Protocol: over standard input and output by "
                    "default, as apps start it from their settings, or over HTTP with --http.",
    )
    served.add_argument("--root", help="the folder the tools may read (default: the folder the server starts "
                                       "in, or your home folder when that is the top of the disk)")
    served.add_argument("--http", action="store_true", help="serve Streamable HTTP at /mcp instead of stdio")
    served.add_argument("--host", default="127.0.0.1", help="the address to listen on with --http "
                                                            "(default: 127.0.0.1)")
    served.add_argument("--port", type=int, default=8000, help="the port with --http (default: 8000)")
    served.add_argument("--allow-origin", action="append", metavar="ORIGIN",
                        help="a web page origin allowed to call the HTTP server (repeatable)")

    computed = subcommands.add_parser(
        "compute", help="compute an expression or solve equations exactly, with every step",
        description="Compute an arithmetic expression exactly: fractions stay fractions, every step is "
                    "shown, and an expression that reads two ways is refused with both readings. It knows "
                    "+ - * / // ^ mod ! %, parentheses, sqrt, root, abs, floor, ceil, round, gcd, lcm, comb, "
                    "perm, min, max, exp, ln, log10, log2, log(x, base), pi and e. With an = sign it solves "
                    "linear equations in one-letter unknowns (several separated by ;), and checks the "
                    "answer by putting it back; an expression with letters is multiplied out.",
    )
    computed.add_argument("expression", nargs="+", help='the expression, such as "2^10 + 3/4"')
    computed.add_argument("--digits", type=int, default=20,
                          help="significant digits for a result that is not exact (default: 20)")
    computed.add_argument("--json", action="store_true", help="machine-readable result")

    subcommands.add_parser("samples", help="list the bundled datasets")
    return parser


def _moved(argv: list[str]) -> int:
    """Hand a command that moved to ADA Lab to it when it is installed, or say how to get it."""
    try:
        from ada_lab.cli import main as lab_main  # noqa: PLC0415
    except ImportError:
        print(
            f"`ada {argv[0]}` is in ADA Lab now, with the research workflow, the modelling mode and the "
            f"browser app:\n    pip install ada-lab\n"
            f"then run `ada-lab {argv[0]}` (or `ada {argv[0]}` again).",
            file=sys.stderr,
        )
        return EXIT_ERROR
    return lab_main(argv, prog="ada")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in MOVED_COMMANDS:
        return _moved(argv)
    if argv and argv[0] == "compute":
        argv = _compute_arguments(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_QUIET
    handler = {
        "analyze": _analyze, "schema": _schema, "ask": _ask, "claim": _claim, "scan": _scan, "check": _check,
        "skill": _skill, "tools": _tools, "chat": _chat, "mcp": _mcp, "compute": _compute,
        "samples": _samples,
    }[args.command]
    try:
        return handler(args)
    except AdaError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
