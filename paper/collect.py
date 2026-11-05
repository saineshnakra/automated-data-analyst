"""Gather a benchmark run's results files into one file for the paper.

    python paper/collect.py RESULTS_DIR paper/runs/RUN --dataset-model-from-name

Every line of every ``*.jsonl`` file in RESULTS_DIR is one record as gsm8k.py or math500.py wrote it. The
records go into RUN/results.jsonl.gz, sorted, with ``dataset`` and ``model`` on each; gsm8k.py's files carry
neither, so with ``--dataset-model-from-name`` they are read from the file's name, as the benchmark
workflows and kaggle_ladder.py name them: ``gsm8k-qwen2.5-1.5b-3.jsonl`` is GSM8K, qwen2.5:1.5b, shard 3.
The file is the same, byte for byte, each time it is made from the same records.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

DATASETS = ("gsm8k", "gsmhard", "math500")
#: The model names the files were made from; a file name has ":" and "/" written as "-".
MODELS = ("smollm2:135m", "gemma3:270m", "smollm2:360m", "qwen2.5:0.5b", "qwen3:0.6b", "gemma3:1b",
          "llama3.2:1b", "qwen2.5:1.5b", "qwen2-math:1.5b", "qwen3:1.7b", "smollm2:1.7b", "llama3.2:3b",
          "qwen2.5:3b", "phi4-mini", "gemma3:4b", "qwen3:4b", "qwen3:4b-instruct")


def named(path: Path) -> tuple[str, str]:
    """The data set and the model a results file's name gives."""
    stem = re.sub(r"-\d+$", "", path.stem)
    dataset = next((name for name in DATASETS if stem.startswith(name + "-")), None)
    if dataset is None:
        raise ValueError(f"{path.name} does not start with a data set name ({', '.join(DATASETS)})")
    slug = stem[len(dataset) + 1:]
    model = next((model for model in MODELS if model.replace(":", "-").replace("/", "-") == slug), None)
    if model is None:
        raise ValueError(f"{path.name}: no known model is written {slug}")
    return dataset, model


def key(record: dict) -> tuple:
    """Records sorted by data set, model, problem, condition and sample."""
    problem = record.get("index", record.get("id"))
    return (record["dataset"], record["model"], str(problem).zfill(8), record["condition"],
            record.get("sample", 0))


def collect(source: Path, target: Path, from_name: bool) -> int:
    records = []
    for path in sorted(source.rglob("*.jsonl")):
        dataset, model = named(path) if from_name else (None, None)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            record.setdefault("dataset", dataset or ("math500" if "subject" in record else None))
            record.setdefault("model", model)
            if record["dataset"] is None or record["model"] is None:
                raise ValueError(f"{path.name}: a record without a data set or model; use "
                                 "--dataset-model-from-name")
            records.append(record)
    # A record made again later, as when a run picks up and asks again, takes the place of the one before.
    latest = {key(record): record for record in records}
    target.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(latest[name], sort_keys=True) + "\n" for name in sorted(latest))
    with open(target / "results.jsonl.gz", "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0,
                                                                        filename="") as handle:
        handle.write(text.encode("utf-8"))
    return len(latest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="a folder of results files")
    parser.add_argument("target", type=Path, help="the run's folder in paper/runs")
    parser.add_argument("--dataset-model-from-name", action="store_true",
                        help="read the data set and model from each file's name")
    args = parser.parse_args()
    count = collect(args.source, args.target, args.dataset_model_from_name)
    print(f"{count} records in {args.target / 'results.jsonl.gz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
