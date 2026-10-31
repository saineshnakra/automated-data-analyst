"""Score ADA's modelling mode on tabular Kaggle competitions; submit only when it should reach the top 10%.

    KAGGLE_API_TOKEN=... python -I run.py ADA_CHECKOUT CACHE out.json [--only=a,b] [--auto] [--submit|--first]

For each competition in competitions.yml this downloads the data (the account
must have accepted the competition's rules), runs ``ada.model`` on it, reads
the leaderboard and works out the score at the top 10%. A submission is sent
only with ``--submit`` and only when every gate passes:

- every check on the submission passes and the model gave no warning;
- the competition has a scored submission of ADA's, and the cross-validated
  score moved by that submission's gap between its leaderboard and
  cross-validated scores beats the top-10% bar by at least twice the larger
  of the folds' standard error (the spread over the square root of the folds)
  and the test wobble: how much the score moves between test sets the size of
  the public leaderboard's share of the test rows, from bootstrap samples of
  the out-of-fold predictions. Test rows can be harder than the training
  rows: on Playground S5E12 a cross-validated AUC of 0.731 scored 0.699, below
  a bar of 0.704 it seemed to clear;
- the competition got no submission in the last 24 hours, and no competition
  got one in the last 2 hours (``submissions.json`` in the cache records them).

With ``--first`` a competition that has never had a submission gets one when
the checks pass and the model gave no warning, whatever its score, so every
competition has a leaderboard score to compare with the cross-validated one.

With ``--auto`` the model is ``ada.model(..., auto=True)``, the wider search
of ``ada auto``.

Submissions go through the ``kaggle`` package. The token is read from the
environment only and never written anywhere.
"""

import io
import json
import math
import os
import re
import sys
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

API = "https://www.kaggle.com/api/v1"
HERE = Path(__file__).parent
DAY, GAP = 24 * 3600, 2 * 3600


def request(path: str, method: str = "GET", data: bytes | None = None, content_type: str | None = None,
            url: str | None = None) -> bytes:
    headers = {"Authorization": f"Bearer {os.environ['KAGGLE_API_TOKEN']}"} if url is None else {}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url or f"{API}/{path}", data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as response:
        return response.read()


def download(slug: str, cache: Path) -> Path:
    folder = cache / slug
    if not (folder / ".done").exists():
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(request(f"competitions/data/download-all/{slug}"))) as archive:
            archive.extractall(folder)
        (folder / ".done").write_text("")
    return folder


def bar(slug: str, higher: bool, cache: Path) -> tuple[float, int]:
    """The leaderboard score at the top 10%, and how many teams the leaderboard holds."""
    path = cache / f"{slug}-leaderboard.zip"
    if not path.exists():
        path.write_bytes(request(f"competitions/{slug}/leaderboard/download"))
    with zipfile.ZipFile(path) as archive:
        board = pd.read_csv(archive.open(archive.namelist()[0]))
    scores = board["Score"].dropna()
    return float(scores.quantile(0.9 if higher else 0.1)), len(scores)


def percentile(score: float, higher: bool, slug: str, cache: Path) -> float:
    """The share of teams a score beats or ties, as a percentile from the top (10 is the top 10%)."""
    with zipfile.ZipFile(cache / f"{slug}-leaderboard.zip") as archive:
        scores = pd.read_csv(archive.open(archive.namelist()[0]))["Score"].dropna()
    ahead = (scores > score).mean() if higher else (scores < score).mean()
    return float(100 * ahead)


def submit(slug: str, path: Path, message: str) -> str:
    """Submit a file through the ``kaggle`` package; Kaggle's REST paths for uploads are not public."""
    from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

    api = KaggleApi()
    api.authenticate()
    return str(api.competition_submit(str(path), message, slug))


def calibration(slug: str) -> tuple[float, float] | None:
    """The cross-validated and public scores of ADA's latest scored submission, read from its description."""
    from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

    api = KaggleApi()
    api.authenticate()
    for sent in api.competition_submissions(slug):
        said = re.match(r"ada model.*\bcv (-?[0-9.]+)", sent.description or "")
        if said and sent.public_score not in (None, ""):
            return float(said.group(1)), float(sent.public_score)
    return None


def allowed(slug: str, log: list[dict]) -> str | None:
    """Why a submission must wait, or None."""
    now = time.time()
    if any(entry["slug"] == slug and now - entry["at"] < DAY for entry in log):
        return "this competition got a submission in the last 24 hours"
    if any(now - entry["at"] < GAP for entry in log):
        return "a competition got a submission in the last 2 hours"
    return None


def main() -> None:
    checkout, cache, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    only = next((arg.split("=", 1)[1].split(",") for arg in sys.argv if arg.startswith("--only=")), None)
    send, first, auto = "--submit" in sys.argv, "--first" in sys.argv, "--auto" in sys.argv
    sys.path.insert(0, str(checkout))
    import ada.modelling as modelling  # noqa: PLC0415

    spec = yaml.safe_load((HERE / "competitions.yml").read_text())["competitions"]
    log_path = cache / "submissions.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    results = json.loads(out.read_text()) if out.exists() else {}
    for item in spec:
        slug = item["slug"]
        if only and slug not in only:
            continue
        metric = modelling.metric_named(item["metric"])
        started = time.time()
        try:
            folder = download(slug, cache)
        except Exception as error:  # noqa: BLE001 - reported per competition
            results[slug] = {"error": f"download: {error}"}
            print(f"{slug}: could not download ({error})")
            continue
        report = modelling.model(folder, metric=item["metric"], auto=auto)
        seconds = round(time.time() - started)
        if not isinstance(report, modelling.ModelReport):
            results[slug] = {"refused": report.message}
            print(f"{slug}: refused: {report.message}")
            continue
        top, teams = bar(slug, metric.higher, cache)
        public = int(item.get("public_share", 0.5) * report.rows_test)
        wobble = report.test_wobble(public)
        error = max(report.spread / math.sqrt(len(report.folds)), wobble)
        scored = calibration(slug)
        offset = scored[1] - scored[0] if scored else 0.0
        expected = report.score + offset
        margin = (expected - top) if metric.higher else (top - expected)
        failed = [check.name for check in report.checks if not check.passed]
        gate = []
        if scored is None:
            gate.append("no scored submission yet to show how the cross-validated score maps to the "
                        "leaderboard; send one with --first")
        if failed:
            gate.append(f"checks failed: {', '.join(failed)}")
        if report.warnings:
            gate.append(f"{len(report.warnings)} warnings")
        if scored is not None and margin < 2 * error:
            gate.append(f"cross-validated {report.score:.5f}, {expected:.5f} after the last submission's "
                        f"leaderboard gap of {offset:+.5f}, is not clear of the top-10% bar {top:.5f} by "
                        f"twice its uncertainty {error:.5f} (test wobble {wobble:.5f} on {public:,} rows)")
        submission = cache / f"{slug}-submission.csv"
        report.write(submission)
        entry = {
            "metric": metric.name, "cv": report.score, "sd": report.spread,
            "leaderboard_gap": offset if scored else None,
            "bar_top10": top, "teams": teams, "wobble": wobble,
            "cv_percentile": percentile(report.score, metric.higher, slug, cache), "seconds": seconds,
            "gate": gate or ["passed"], "warnings": list(report.warnings), "auto": auto,
        }
        print(f"{slug} {metric.name} cv {report.score:.5f} (sd {report.spread:.5f}) "
              f"bar {top:.5f} -> about top {entry['cv_percentile']:.1f}%; gate: {'; '.join(entry['gate'])}")
        sent_before = any(logged["slug"] == slug for logged in log)
        firsts = first and not failed and not report.warnings and not sent_before
        if send and not gate or firsts:
            # A first submission does not wait: Kaggle limits each competition, not the account.
            wait = None if firsts else allowed(slug, log)
            if wait:
                entry["submitted"] = f"held: {wait}"
            else:
                message = (f"ada {'auto' if auto else 'model'}, {metric.name} cv {report.score:.5f} "
                           f"sd {report.spread:.5f}")
                entry["submitted"] = submit(slug, submission, message)
                log.append({"slug": slug, "at": time.time(),
                            "when": datetime.now(UTC).isoformat(timespec="seconds")})
                log_path.write_text(json.dumps(log, indent=2))
        results[slug] = entry
        out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
