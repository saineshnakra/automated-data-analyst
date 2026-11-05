"""GSM8K and GSM-Hard on Kaggle's two T4 GPUs: one Ollama server on each GPU, the models split between them,
and every shard's results left in /kaggle/working/results.

Kaggle runs a script kernel as one file, so the settings are in CONFIG below; this is the run recorded in
benchmarks/README.md. To run it, put this file in a folder with a kernel-metadata.json that names it as the
code file, with "enable_gpu": true and "enable_internet": true, then:

    kaggle kernels push -p <folder> --accelerator NvidiaTeslaT4
    kaggle kernels output <user>/<kernel> -p <out> --file-pattern 'results/.*'

Each results file is gsm8k.py's output for one model, data set and shard; summary.py reads it.
"""

import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

#: The run: the commit of this repository to test, the data, the models, the conditions, the first N
#: problems and how many shards each data set is split into.
CONFIG = {
    "commit": "c4b7e370de55e6044c2f10648bdc7bf08f57bc4c",
    "datasets": ["gsm8k", "gsmhard"],
    "models": ["qwen3:4b-instruct", "gemma3:4b", "phi4-mini", "qwen2.5:3b", "llama3.2:3b", "smollm2:1.7b",
               "qwen3:1.7b", "qwen2-math:1.5b", "qwen2.5:1.5b", "llama3.2:1b", "gemma3:1b", "qwen3:0.6b",
               "qwen2.5:0.5b"],
    "conditions": "none,shots,check,solve",
    "problems": 300,
    "parallel": 6,
}
WORK = Path("/kaggle/working")
REPO = Path("/tmp/repo")
GSM8K = ("https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
         "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14")
GSMHARD = ("https://raw.githubusercontent.com/reasoning-machines/pal/main/datasets/gsmhardv2.jsonl",
           "2032b0188a49ac0ec38cbe1e695e7d62d6dddc44bbb64f920728aca710e9ddef")


def sh(command: str, env: dict | None = None) -> None:
    print("+", command, flush=True)
    subprocess.run(command, shell=True, check=True, env=env)


def fetched(url_and_hash: tuple[str, str], path: Path) -> Path:
    url, digest = url_and_hash
    sh(f"curl -fsSL -o {path} {url}")
    sh(f"echo '{digest}  {path}' | sha256sum -c")
    return path


def serve(gpu: int, port: int, parallel: int) -> None:
    log = open(WORK / f"ollama-{gpu}.log", "w")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OLLAMA_HOST": f"127.0.0.1:{port}",
           "OLLAMA_NUM_PARALLEL": str(parallel), "OLLAMA_KEEP_ALIVE": "-1"}
    subprocess.Popen(["ollama", "serve"], stdout=log, stderr=subprocess.STDOUT, env=env)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/tags", timeout=5)
            return
        except OSError:
            time.sleep(2)


def ladder(models: list[str], port: int, datasets: dict[str, Path], parallel: int) -> None:
    results = WORK / "results"
    env = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{port}"}
    for model in models:
        try:
            sh(f"ollama pull {model} > /dev/null", env)
        except subprocess.CalledProcessError:
            print("could not pull", model, flush=True)
            continue
        started = time.time()
        slug = model.replace(":", "-").replace("/", "-")
        jobs = []
        for name, data in datasets.items():
            for part in range(parallel):
                out = results / f"{name}-{slug}-{part}.jsonl"
                command = [sys.executable, str(REPO / "benchmarks/gsm8k.py"), str(data), "--model", model,
                           "--base-url", f"http://127.0.0.1:{port}", "--condition", CONFIG["conditions"],
                           "--problems", str(CONFIG["problems"]), "--shard", f"{part}/{parallel}",
                           "--out", str(out)]
                log = open(f"{out}.log", "w")
                jobs.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        for job in jobs:
            job.wait()
        print(f"{model}: {time.time() - started:.0f} s", flush=True)
        subprocess.run(["df", "-h", "/root"], check=False)


def main() -> int:
    parallel = int(CONFIG.get("parallel", 4))
    sh("command -v zstd || (apt-get update -qq && apt-get install -y -qq zstd)")
    sh("curl -fsSL https://ollama.com/install.sh | sh")
    listed = subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout
    gpus = int(listed or 1)
    ports = [11434 + gpu for gpu in range(max(1, min(gpus, 2)))]
    for gpu, port in enumerate(ports):
        serve(gpu, port, parallel)
    sh(f"git clone -q https://github.com/saineshnakra/automated-data-analyst.git {REPO} && "
       f"git -C {REPO} checkout -q {CONFIG['commit']}")
    sh(f"pip install -q -e {REPO}")
    datasets = {name: fetched(source, WORK / f"{name}.jsonl")
                for name, source in (("gsm8k", GSM8K), ("gsmhard", GSMHARD)) if name in CONFIG["datasets"]}
    (WORK / "results").mkdir(exist_ok=True)
    models = CONFIG["models"]
    # The models alternate between the GPUs, so each GPU has a mix of sizes.
    threads = [threading.Thread(target=ladder, args=(models[index::len(ports)], port, datasets, parallel))
               for index, port in enumerate(ports)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    sh("nvidia-smi || true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
