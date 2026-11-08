"""Run the checks of .github/workflows/ci.yml on this computer, before a merge.

    python tools/check.py
    python tools/check.py python3.11 python3.12 python3.13

For each Python named (this one when none is), it makes a fresh virtual environment, installs
requirements-dev.txt, and runs ``ruff check .``, the unit suite and ``compileall``. Then, with the first
Python, it builds the wheel, installs it in another fresh environment without the app extras and runs the
installed command: ``ada compute``, a claim, a question, a folder scan, the MCP server and a command that
moved to ADA Lab. It stops at the first check that fails and exits 1. The checks are the ones in ci.yml, so a
change to one belongs in the other.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = "Scripts" if os.name == "nt" else "bin"


class Failed(Exception):
    pass


def run(command: list[str], cwd: Path = ROOT, expect: int = 0, stdin: str | None = None,
        quiet: bool = False) -> subprocess.CompletedProcess:
    """Run a command and return it, or raise Failed when it exits with another code than ``expect``."""
    print("+", " ".join(command), flush=True)
    done = subprocess.run(command, cwd=cwd, input=stdin, text=True, capture_output=True)
    if done.returncode != expect:
        print(done.stdout[-4000:], done.stderr[-4000:], sep="\n")
        raise Failed(f"{command[0]} exited with {done.returncode}, not {expect}")
    if not quiet and done.stdout.strip():
        print(done.stdout.strip()[-2000:])
    return done


def environment(python: str, folder: Path) -> Path:
    """A fresh virtual environment of ``python`` in ``folder``; its own python."""
    run([python, "-m", "venv", str(folder)])
    return folder / BIN / ("python.exe" if os.name == "nt" else "python")


def lint_and_test(python: str, scratch: Path) -> None:
    """Lint, the unit suite and compileall, with ``python``'s own environment."""
    own = environment(python, scratch / "test")
    run([str(own), "-m", "pip", "install", "-q", "-r", "requirements-dev.txt"])
    run([str(own), "-m", "ruff", "check", "."])
    found = run([str(own), "-m", "unittest", "discover", "-s", "tests"], quiet=True)
    print(*[line for line in found.stderr.splitlines() if line.startswith("Ran ")], sep="\n")
    run([str(own), "-m", "compileall", "-q", "app.py", "ada", "tools", "tests"])


def package(python: str, scratch: Path) -> None:
    """Build the wheel, install it without the app extras and run the installed command."""
    builder = environment(python, scratch / "build")
    run([str(builder), "-m", "pip", "install", "-q", "build"])
    run([str(builder), "-m", "build", "--wheel", "--outdir", str(scratch / "dist")], quiet=True)
    installed = environment(python, scratch / "installed")
    run([str(installed), "-m", "pip", "install", "-q", *map(str, (scratch / "dist").glob("*.whl"))])
    command = installed.parent / ("ada.exe" if os.name == "nt" else "ada")
    work = scratch / "work"
    work.mkdir()

    def ada(*arguments: str, expect: int = 0, stdin: str | None = None) -> str:
        return run([str(command), *arguments], cwd=work, expect=expect, stdin=stdin, quiet=True).stdout

    def check(condition: bool, what: str) -> None:
        if not condition:
            raise Failed(what)
        print("ok:", what)

    check(ada("--version").startswith("ada "), "ada --version")
    ada("analyze", "-")
    skill = ada("skill", "--print")
    check("name: ada" in skill.splitlines() and "ada claim" in skill, "ada skill --print")
    check('"y": "4"' in ada("compute", "x + y = 10; x - y = 2", "--json"), "ada compute solves a system")
    check('"kind": "several"' in ada("compute", "x*y = 6; x + y = 5", "--json"),
          "ada compute finds two answers")
    run([str(installed), "-c", "import ada, sys; ada.analyze(ada.sample()); "
         "assert not {'streamlit', 'matplotlib'} & set(sys.modules)"], cwd=work)
    (work / "orders.csv").write_text("order_id,amount\na,10.0\na,5.0\nb,20.0\nc,30.0\n")
    (work / "regions.csv").write_text("order_id,region\na,North\nb,South\nc,North\n")
    check(json.loads(ada("claim", "orders.csv", "The total amount is 65.", "--json"))["status"] == "verified",
          "ada claim verifies a true claim")
    ada("claim", "orders.csv", "The total amount is 70.", expect=1)
    print("ok: ada claim exits 1 on a false claim")
    check(json.loads(ada("ask", "orders.csv", "total amount", "--json"))["result"]["value"] == 65.0,
          "ada ask answers")
    check(json.loads(ada("scan", ".", "--json"))["links"][0]["to"] == "regions.order_id",
          "ada scan links the files")
    replies = [json.loads(line) for line in ada("mcp", stdin=(
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}\n'
        '{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n')).splitlines() if line.strip()]
    names = [tool["name"] for tool in replies[1]["result"]["tools"]]
    check(names[0] == "ada_ask" and "ada_compute" in names, "ada mcp lists the tools")
    longer = "automated-data-analyst.exe" if os.name == "nt" else "automated-data-analyst"
    run([str(installed.parent / longer), "--version"], cwd=work)
    run([str(installed), "-m", "ada", "--version"], cwd=work)
    moved = run([str(command), "build"], cwd=work, expect=2, quiet=True)
    check("pip install ada-lab" in moved.stderr, "ada build points to ADA Lab")


def main(arguments: list[str]) -> int:
    pythons = arguments or [sys.executable]
    try:
        for python in pythons:
            print(f"== {python}: lint, tests, compileall", flush=True)
            with tempfile.TemporaryDirectory() as scratch:
                lint_and_test(python, Path(scratch))
        print(f"== {pythons[0]}: the installed wheel", flush=True)
        with tempfile.TemporaryDirectory() as scratch:
            package(pythons[0], Path(scratch))
    except Failed as error:
        print(f"FAILED: {error}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
