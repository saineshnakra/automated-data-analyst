"""The ``ada`` command.

For a research study, run these in the folder that holds the data:

    ada init data/measurements.csv     # answer a few questions; writes study.yml
    ada build                          # tables, figures, methods and more in paper-assets/
    ada verify                         # rebuild and check every number still matches
    ada share                          # copy what may be shared into share/

To try it on an example study first: ``ada demo``. ``ada build`` counts the analyses
study.yml asks for, fitted and refused; ``ada build --strict`` exits with 1 when any
was refused, so a manuscript build cannot pass without them. ``ada skill`` installs
the skill that tells AI coding assistants how to use ADA.

For a first look at any table:

    ada analyze measurements.csv
    ada schema measurements.csv
    ada ask orders.csv "which category made the most revenue"       # --json for programs
    ada claim flights.csv "The average dep_delay was 12.64 minutes."   # check a stated number
    ada check metrics.csv --measure signups --frequency W
    ada check rainfall.csv --measure "Rainfall (mm)" --aggregation sum
    ada samples
    ada app

``ada claim`` checks one number someone states about a table and says verified, or
unverified with the reason (see ``ada.claims``). It is not ``ada verify``, which
rebuilds a study's paper assets and compares them, nor ``ada check``, which watches
a measure for unusual periods. It reports through its exit code too:

    0  verified
    1  unverified: the data give a different number, or a check on the analysis failed
    2  the data could not be read
    3  unverified for another reason: missing context, ambiguous wording, unsupported

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
from .metrics import AGGREGATION_CHOICES, metric_for
from .nlq import ANSWER_SCHEMA_VERSION

EXIT_QUIET = 0
EXIT_FLAGGED = 1
EXIT_ERROR = 2
EXIT_COULD_NOT_LOOK = 3


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


def _ask(args: argparse.Namespace) -> int:
    answer = _load(args).ask(args.question)
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
    return EXIT_QUIET


def _claim(args: argparse.Namespace) -> int:
    if args.claim is None and (args.question is None or args.value is None):
        print("error: give the claim as a sentence, or both --question and --value.", file=sys.stderr)
        return EXIT_ERROR
    check = check_claim(args.file, args.claim, question=args.question, value=args.value, sheet=args.sheet)
    print(json.dumps(check.to_dict(), indent=2) if args.json else check)
    if check.verified:
        return EXIT_QUIET
    if check.reason == "failed_check":
        return EXIT_FLAGGED
    return EXIT_ERROR if check.reason == "inaccessible_data" else EXIT_COULD_NOT_LOOK


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


def _init(args: argparse.Namespace) -> int:
    from .research import init  # noqa: PLC0415

    try:
        path = init(args.file, study_path=args.study, sheet=args.sheet, interactive=not args.yes,
                    overwrite=args.force)
    except (FileNotFoundError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR
    print(f"\nWrote {path}. Read it through, fill in anything blank, then run `ada build`.")
    return EXIT_QUIET


def _build(args: argparse.Namespace) -> int:
    from .research import build  # noqa: PLC0415

    built = build(args.study, out=Path(args.out) if args.out else None)
    checks = sum(1 for finding in built.findings if finding.status == "check")
    print(f"Built {len(built.assets)} assets in {built.out}")
    for asset in built.assets:
        if asset.section in ("main", "supplement"):
            print(f"  {asset.number:12} {asset.title}")
    # Counted apart from the notes, so a build can't look complete while an analysis never ran.
    requested, refused = len(built.results), built.refused
    if requested:
        fitted = requested - len(refused)
        print(f"\nAnalyses: {requested} requested, {fitted} fitted, {len(refused)} refused.")
    else:
        print("\nAnalyses: none requested in the study file.")
    for ident, reason in refused.items():
        print(f"  refused  {ident}: {reason}")
    if checks:
        print(f"\nThe data audit lists {checks} item(s) to check: {built.out / 'audit' / 'data-audit.md'}")
    changed = [change for change in built.changes if change.status != "new"]
    if changed:
        print(f"\n{len(changed)} asset(s) changed since the last build; see {built.out / 'CHANGES.md'}")
    listed = {f"Analysis {ident} was not fitted: {reason}" for ident, reason in refused.items()}
    for warning in built.warnings:
        if warning not in listed:
            print(f"note: {warning}")
    if args.strict and refused:
        print(
            f"error: --strict: {len(refused)} of {requested} requested analyses "
            f"{'was' if len(refused) == 1 else 'were'} refused ({', '.join(refused)}). "
            "The assets were written, but without the refused analyses.",
            file=sys.stderr,
        )
        return EXIT_FLAGGED
    return EXIT_QUIET


def _verify(args: argparse.Namespace) -> int:
    from .research.verify import verify  # noqa: PLC0415

    report = verify(args.study, rtol=args.rtol, atol=args.atol, out=args.out)
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.text())
    return EXIT_QUIET if report.ok else EXIT_FLAGGED


def _preview(args: argparse.Namespace) -> int:
    from .research.package import preview  # noqa: PLC0415

    print(preview(args.study))
    return EXIT_QUIET


def _share(args: argparse.Namespace) -> int:
    from .research.package import share  # noqa: PLC0415

    shared = share(args.study, out=args.out)
    print(shared.text())
    return EXIT_QUIET if shared.ok else EXIT_FLAGGED


def _demo(args: argparse.Namespace) -> int:
    import shlex  # noqa: PLC0415

    from .research.example import write_example  # noqa: PLC0415

    try:
        files = write_example(args.folder, overwrite=args.force)
    except FileExistsError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR
    folder = Path(args.folder)
    print(f"Wrote an example study to {folder}/: sixteen mice on two diets, weighed weekly (simulated data).")
    for path in files:
        print(f"  {path.relative_to(folder).as_posix()}")
    print(f"\nNext:\n    cd {shlex.quote(str(folder))} && ada build\n"
          "then open paper-assets/README.md. `ada preview` lists what may be shared, and `ada share` "
          "copies it.")
    return EXIT_QUIET


def _status(_: argparse.Namespace) -> int:
    """With no command: say where this folder stands and what to run next."""
    study = Path("study.yml")
    if not study.is_file():
        print("No study.yml here. To make paper assets for a study, run this in its folder:\n"
              "    ada init path/to/data.csv\n"
              "then `ada build`. To try it on an example study: `ada demo`.\n"
              "For a quick look at any table: `ada analyze file.csv`.\n"
              "`ada --help` lists every command.")
        return EXIT_QUIET
    from .research import load  # noqa: PLC0415
    from .research.provenance import read_manifest  # noqa: PLC0415

    loaded = load(study)
    record = read_manifest(loaded.out_dir)
    print(f"{loaded.title}: {loaded.data_path}, {loaded.design} design, {len(loaded.analyses)} analyses")
    if record is None:
        print("Not built yet. Run `ada build`.")
    else:
        print(f"{len(record['assets'])} assets in {loaded.output.dir}/ from ADA {record['ada']}. "
              "Run `ada verify` to check they are current, or `ada build` to rebuild.")
    return EXIT_QUIET


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


def _samples(_: argparse.Namespace) -> int:
    for name, path in sorted(samples().items()):
        print(f"{name:22} {path}")
    return EXIT_QUIET


def _app(_: argparse.Namespace) -> int:
    """Launch the Streamlit app from a source checkout (needs the [app] extra)."""
    entry = Path(__file__).resolve().parent.parent / "app.py"
    if not entry.is_file():
        print(
            "The app runs from a source checkout:\n"
            "    git clone https://github.com/saineshnakra/automated-data-analyst\n"
            "    cd automated-data-analyst && pip install -e '.[app]' && ada app",
            file=sys.stderr,
        )
        return EXIT_ERROR
    try:
        from streamlit.web import cli as streamlit_cli  # noqa: PLC0415
    except ImportError:
        print("The app needs Streamlit: pip install -e '.[app]'", file=sys.stderr)
        return EXIT_ERROR
    sys.argv = ["streamlit", "run", str(entry)]
    streamlit_cli.main()
    return EXIT_QUIET


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ada",
        description="Paper assets for a study (init, build, verify) and a first look at any table (analyze).",
    )
    parser.add_argument("--version", action="version", version=f"ada {__version__}")
    subcommands = parser.add_subparsers(dest="command", metavar="COMMAND")

    def research_commands(group) -> None:
        init = group.add_parser(
            "init", help="describe a study: writes study.yml from the data and your answers"
        )
        init.add_argument("file", help="the study's data file (CSV, TSV or Excel)")
        init.add_argument("--sheet", help="the worksheet to read from an Excel file")
        init.add_argument("--study", default="study.yml", help="where to write the study file")
        init.add_argument("--yes", action="store_true", help="accept every suggestion without asking")
        init.add_argument("--force", action="store_true", help="replace an existing study file")
        built = group.add_parser("build", help="make the paper assets: tables, figures, methods and records")
        built.add_argument("--study", default="study.yml", help="the study file")
        built.add_argument("--out", help="the output folder (default: output.dir in the study file)")
        built.add_argument(
            "--strict", action="store_true", help="exit with status 1 if any analysis in the study is refused"
        )
        check = group.add_parser("verify", help="rebuild in a scratch folder and check every asset matches")
        check.add_argument("--study", default="study.yml", help="the study file")
        check.add_argument("--out", help="the output folder to check (default: output.dir in the study file)")
        check.add_argument("--rtol", type=float, default=1e-9, help="relative tolerance for numbers (1e-9)")
        check.add_argument("--atol", type=float, default=1e-12, help="absolute tolerance for numbers (1e-12)")
        check.add_argument("--json", action="store_true", help="machine-readable output")
        shown = group.add_parser("preview", help="list what would be shared, and anything that should not be")
        shown.add_argument("--study", default="study.yml", help="the study file")
        shared = group.add_parser("share", help="copy the files that may be shared into share/")
        shared.add_argument("--study", default="study.yml", help="the study file")
        shared.add_argument("--out", help="the folder to write (default: share/ beside the study file)")
        demo = group.add_parser("demo", help="write a small example study to try ada build on")
        demo.add_argument("folder", nargs="?", default="ada-demo",
                          help="where to write it (default: ada-demo)")
        demo.add_argument("--force", action="store_true", help="write into a folder that is not empty")

    research_commands(subcommands)
    research = subcommands.add_parser("research", help="the same research commands, under one name")
    research_commands(research.add_subparsers(dest="research_command", metavar="COMMAND", required=True))

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
    asked = with_source("ask", "ask a question in plain English")
    asked.add_argument("question")
    asked.add_argument("--json", action="store_true", help="machine-readable answer with its evidence")
    claimed = subcommands.add_parser(
        "claim", help="check a number stated about a table: verified, or unverified with the reason"
    )
    claimed.add_argument("file", help="the CSV, TSV or Excel file the claim is about")
    claimed.add_argument("claim", nargs="?", help='the claim, such as "The average delay was 12.6 minutes."')
    claimed.add_argument("--question", help="the calculation, when it is known: \"average dep_delay\"")
    claimed.add_argument("--value", help="the number claimed, with --question; its decimals set precision")
    claimed.add_argument("--sheet", help="the worksheet to read from a workbook (default: the first)")
    claimed.add_argument("--json", action="store_true", help="machine-readable result")
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

    subcommands.add_parser("samples", help="list the bundled datasets")
    subcommands.add_parser("app", help="launch the browser app (from a source checkout)")
    return parser


def main(argv: list[str] | None = None) -> int:
    from .research.study import StudyError  # noqa: PLC0415

    args = build_parser().parse_args(argv)
    command = getattr(args, "research_command", None) or args.command
    handler = {
        "init": _init, "build": _build, "verify": _verify, "preview": _preview,
        "share": _share, "demo": _demo,
        "analyze": _analyze, "schema": _schema, "ask": _ask, "claim": _claim,
        "check": _check, "skill": _skill, "samples": _samples, "app": _app, None: _status,
    }[command]
    try:
        return handler(args)
    except (AdaError, StudyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
