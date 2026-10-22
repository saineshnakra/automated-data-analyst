"""Where each asset came from, and what changed since the last build.

Every build writes ``provenance.json``: for each asset, its files and their
sha256, the input file's checksum, the rows it used and the exclusion steps
behind them, the parts of ``study.yml`` it depends on (hashed), the function
that made it, the calculation in words, references, and the package versions.
The next build compares itself with that record and writes ``CHANGES.md``,
naming each asset that changed and why.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from .study import Study, sha256_of

MANIFEST = "provenance.json"
CHANGES = "CHANGES.md"
FORMAT = 1

#: Distributions whose versions can change a number or a figure.
PACKAGES = (
    "automated-data-analyst",
    "pandas",
    "numpy",
    "scipy",
    "statsmodels",
    "matplotlib",
    "python-docx",
    "PyYAML",
    "ruptures",
    "openpyxl",
    "python-calamine",
)
#: Files compared byte for byte by ``ada verify``. Others (images, Word files) embed
#: library and font details, so only their presence and their source data are compared.
TEXT_SUFFIXES = {".csv", ".md", ".tex", ".txt", ".json", ".cff", ".yml", ".yaml", ".sha256", ".py", ".sh"}


@dataclass
class Asset:
    id: str
    kind: str
    title: str
    files: list[Path] = field(default_factory=list)
    #: Parts of study.yml the asset is built from, such as "processing" or "analyses.primary".
    depends: tuple[str, ...] = ()
    calculation: str = ""
    references: tuple[str, ...] = ()
    code: str = ""
    #: Rows behind the asset: {"used": n, "excluded": n, "steps": [numbers]}.
    rows: dict[str, Any] = field(default_factory=dict)
    #: Headline numbers, so the change report can show old → new.
    numbers: dict[str, Any] = field(default_factory=dict)
    #: "Table 2" or "Figure 1" once numbered for the paper.
    number: str = ""
    caption: str = ""
    #: Whether the files list individual rows of the data.
    row_level: bool = False
    #: "main", "supplement" or "record": where the asset belongs in a paper.
    section: str = "main"


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def git_state(folder: Path) -> dict[str, Any]:
    """The commit and whether the working tree has uncommitted changes, when ``folder`` is in git."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(folder), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(folder), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {}
    return {"commit": commit, "uncommitted_changes": bool(dirty)}


def manifest(
    study: Study, out: Path, assets: list[Asset], processed, custom: dict[str, str]
) -> dict[str, Any]:
    from .. import __version__  # noqa: PLC0415

    def relative(path: Path) -> str:
        return path.resolve().relative_to(out).as_posix()

    records = []
    for record in processed.records:
        records.append(
            {
                "step": record.number,
                "kind": record.kind,
                "description": record.description,
                "reason": record.reason,
                "rows_before": record.rows_before,
                "rows_after": record.rows_after,
                "values_changed": record.values_changed,
            }
        )
    entries = {}
    for asset in assets:
        entries[asset.id] = {
            "kind": asset.kind,
            "title": asset.title,
            "number": asset.number,
            "section": asset.section,
            "caption": asset.caption,
            "files": {relative(path): sha256_of(path) for path in asset.files if path.is_file()},
            "depends_on": list(asset.depends),
            "config_sha256": study.section_hash(*asset.depends),
            "rows": asset.rows,
            "calculation": asset.calculation,
            "references": list(asset.references),
            "code": asset.code,
            "numbers": _plain(asset.numbers),
            "row_level": asset.row_level,
        }
    return {
        "format": FORMAT,
        "ada": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "packages": package_versions(),
        "study": {"file": study.path.name, "sha256": sha256_of(study.path)},
        "inputs": [
            {
                "path": study.data_path,
                "sheet": study.data_sheet,
                "sha256": processed.sha256,
                "rows": len(processed.original),
                "columns": len(processed.original.columns) - 1,
            }
        ],
        "custom": [{"script": script, "sha256": digest} for script, digest in custom.items()],
        "git": git_state(study.root),
        "processing": records,
        "rows_used": len(processed.data),
        "assets": entries,
    }


def _plain(value):
    """Numbers as plain floats and ints so the record reads the same on every platform."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return float(f"{value:.12g}")
    return value


def read_manifest(out: Path) -> dict[str, Any] | None:
    path = out / MANIFEST
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_manifest(out: Path, record: dict[str, Any]) -> Path:
    path = out / MANIFEST
    path.write_text(json.dumps(record, indent=1, sort_keys=False, default=str) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class Change:
    asset: str
    status: str  # new, removed, changed
    reasons: tuple[str, ...]


def compare(before: dict[str, Any] | None, after: dict[str, Any]) -> list[Change]:
    """Each asset that is new, removed or changed between two records, with the reasons."""
    if before is None:
        return []
    old_assets, new_assets = before.get("assets", {}), after.get("assets", {})
    general = []
    if before.get("inputs", [{}])[0].get("sha256") != after["inputs"][0]["sha256"]:
        general.append("the input data changed")
    if before.get("ada") != after.get("ada"):
        general.append(f"ADA {before.get('ada')} → {after.get('ada')}")
    changed_packages = [
        f"{name} {before.get('packages', {}).get(name)} → {version}"
        for name, version in after.get("packages", {}).items()
        if before.get("packages", {}).get(name) not in (None, version)
    ]
    if changed_packages:
        general.append("packages changed: " + ", ".join(changed_packages))
    if before.get("processing") != after.get("processing"):
        general.append("the processing steps changed")
    changes = []
    for ident, entry in new_assets.items():
        old = old_assets.get(ident)
        if old is None:
            changes.append(Change(ident, "new", ()))
            continue
        files_changed = [
            path
            for path, digest in entry["files"].items()
            if Path(path).suffix in TEXT_SUFFIXES and old.get("files", {}).get(path) != digest
        ]
        reasons = []
        if old.get("config_sha256") != entry["config_sha256"]:
            reasons.append("study.yml changed in " + ", ".join(entry["depends_on"]))
        if old.get("rows") != entry["rows"]:
            reasons.append(_rows_change(old.get("rows", {}), entry["rows"]))
        number_changes = _number_changes(old.get("numbers", {}), entry.get("numbers", {}))
        if number_changes:
            reasons.append(number_changes)
        if files_changed and not reasons:
            reasons.extend(general or ["its files changed"])
        if files_changed or reasons:
            reasons.append("files: " + ", ".join(files_changed) if files_changed else "")
            changes.append(Change(ident, "changed", tuple(reason for reason in reasons if reason)))
    for ident in old_assets:
        if ident not in new_assets:
            changes.append(Change(ident, "removed", ()))
    return changes


def _rows_change(old: dict, new: dict) -> str:
    parts = [
        f"{key} {old.get(key)} → {new.get(key)}"
        for key in sorted(set(old) | set(new))
        if old.get(key) != new.get(key)
    ]
    return "rows: " + "; ".join(parts)


def _number_changes(old: dict, new: dict) -> str:
    parts = []
    for key in new:
        if key in old and old[key] != new[key]:
            parts.append(f"{key} {_short(old[key])} → {_short(new[key])}")
    return "numbers: " + "; ".join(parts) if parts else ""


def _short(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def changes_markdown(changes: list[Change], before: dict | None, after: dict, unchanged: int) -> str:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"## Build of {stamp}", ""]
    if before is None:
        lines += [
            f"First build: {len(after['assets'])} assets from {after['inputs'][0]['path']} "
            f"(sha256 {after['inputs'][0]['sha256'][:12]}…), {after['rows_used']:,} rows used.",
            "",
        ]
        return "\n".join(lines)
    if not changes:
        lines += ["Nothing changed since the previous build.", ""]
        return "\n".join(lines)
    for change in changes:
        if change.status == "changed":
            lines.append(f"- **{change.asset}** changed: " + "; ".join(change.reasons) + ".")
        else:
            lines.append(f"- **{change.asset}** is {change.status}.")
    lines += ["", f"{unchanged} other asset(s) unchanged.", ""]
    return "\n".join(lines)


def write_changes(out: Path, text: str) -> Path:
    """Put the latest build's changes at the top of ``CHANGES.md``, keeping the earlier entries."""
    path = out / CHANGES
    previous = path.read_text(encoding="utf-8") if path.is_file() else ""
    body = previous.split("\n", 2)[2] if previous.startswith("# Changes") else previous
    path.write_text(
        "# Changes\n\n" + text + ("\n" + body.lstrip("\n") if body.strip() else ""), encoding="utf-8"
    )
    return path
