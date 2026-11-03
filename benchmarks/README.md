# Benchmarks

## GSM8K and GSM-Hard with small local models

Does a small model get more grade-school word problems right with ADA? Each
problem is solved three ways, with the same model and settings:

- **Alone**: the model works the problem and ends with "Answer: <number>".
- **With `ada_compute`**: the model may call ADA's arithmetic tool for any
  step. A call it writes out as text instead of making is made for it, as
  `ada chat` does.
- **Alone, then checked**: the model's own answer from the first condition,
  checked by `ada.check_arithmetic`. When an equation in it is wrong, the
  model is told which, with the right value, and answers once more. When
  none is, the answer stands, so this condition only changes answers that
  have a slip in them.

### Data and settings

| | |
|---|---|
| GSM8K | The first 200 problems of `test.jsonl` from the grade-school-math repository, sha256 `3730d312…c39d14`. Answers are whole numbers, matched exactly |
| GSM-Hard | The first 200 problems of `gsmhardv2.jsonl` from the PAL repository: the GSM8K problems with larger numbers, sha256 `2032b018…e9ddef`. Answers are matched to within 0.001 |
| Models | qwen3:1.7b (thinking off), qwen2.5:1.5b and llama3.2:1b, through Ollama's chat API |
| Settings | Temperature 0, seed 0, context 8192 tokens, at most 2048 tokens a reply, at most 8 tool rounds |
| Where | GitHub's `ubuntu-latest` runners, CPU only, one job per model and dataset ([benchmark.yml](../.github/workflows/benchmark.yml)) |
| Code | Workflow run 41361082298, commit bfb0b03 |

### Results

Problems right, of 200:

| Model | Data | Alone | With `ada_compute` | Alone, then checked |
|---|---|---|---|---|
| qwen3:1.7b | GSM8K | 155 (77.5%) | 149 (74.5%) | 158 (79.0%) |
| qwen3:1.7b | GSM-Hard | 99 (49.5%) | 106 (53.0%) | 111 (55.5%) |
| qwen2.5:1.5b | GSM8K | 124 (62.0%) | 133 (66.5%) | 123 (61.5%) |
| qwen2.5:1.5b | GSM-Hard | 67 (33.5%) | 73 (36.5%) | 74 (37.0%) |
| llama3.2:1b | GSM8K | 85 (42.5%) | 23 (11.5%) | 89 (44.5%) |
| llama3.2:1b | GSM-Hard | 27 (13.5%) | 21 (10.5%) | 33 (16.5%) |

Against the model alone, on the same problems: right only with ADA / right
only without, and the exact McNemar test (SciPy's `binomtest` on the
problems where the two differ):

| Model | Data | Checked | p | With `ada_compute` | p |
|---|---|---|---|---|---|
| qwen3:1.7b | GSM8K | 3 / 0 | 0.25 | 12 / 18 | 0.36 |
| qwen3:1.7b | GSM-Hard | 12 / 0 | 0.0005 | 27 / 20 | 0.38 |
| qwen2.5:1.5b | GSM8K | 0 / 1 | 1 | 30 / 21 | 0.26 |
| qwen2.5:1.5b | GSM-Hard | 7 / 0 | 0.016 | 26 / 20 | 0.46 |
| llama3.2:1b | GSM8K | 4 / 0 | 0.13 | 6 / 68 | 2e-14 |
| llama3.2:1b | GSM-Hard | 6 / 0 | 0.031 | 13 / 19 | 0.38 |
| All six | | 32 / 1 | 8e-9 | 114 / 166 | 0.002 |

The equations the models wrote in their own answers, and how many ADA
found wrong:

| Model | GSM8K | GSM-Hard |
|---|---|---|
| qwen3:1.7b | 5 of 371 | 35 of 398 |
| qwen2.5:1.5b | 3 of 284 | 57 of 312 |
| llama3.2:1b | 23 of 442 | 161 of 327 |

### What the results show

- **Checking a model's own arithmetic helps, and almost never hurts.**
  Across the six runs it turned 32 wrong answers right and one right answer
  wrong. It helps most where the arithmetic is hard: on GSM-Hard, 12, 7 and
  6 answers with none lost.
- **On GSM8K, small models rarely slip on arithmetic.** 3 to 23 of 284 to
  442 equations are wrong; most wrong answers come from the reasoning, which
  no arithmetic tool fixes. With 7-digit numbers (GSM-Hard), 9% to 49% of
  equations are wrong.
- **Most flagged answers stay wrong.** Of 191 answers with a flagged
  equation, the second reply fixed 32. Often the slip was not the only
  mistake, or the model kept the old number in later steps.
- **Offering the tool is no better than leaving it out for the two qwen
  models**, and much worse for llama3.2:1b on GSM8K. With a tool offered,
  llama3.2:1b calls it on 198 of 200 problems, 186 times with a single call,
  and then answers with whatever comes back: its last reply has a median of
  12 characters, against 515 when it works alone. 24 of its 218 calls copy
  an example from the tool's description. None of the tool's differences
  for the qwen models passes p < 0.25.

So for a small model, the better use of ADA is to let it reason in its own
words and check the arithmetic afterwards. A model that plans before it
calls a tool may do better with `ada_compute`; these three do not.

### How far to trust these numbers

- **Every flag was read.** Over both runs, ADA checked 3,929 equations in
  2,200 replies and flagged 357 distinct ones. Each was read by hand, and
  each is wrong as written; one is a number the model abandoned partway
  ("= 448,  8559716.55 × 50 + …").
- **Replies repeat closely, not exactly.** Five of the jobs also ran in an
  earlier run, with the same code for the model alone. In four, 190 to 196
  of 200 replies were the same word for word and no answer changed between
  right and wrong. For llama3.2:1b on GSM-Hard, 79 of 200 replies were the
  same and 7 answers changed. The runners are not all the same machine.
- **200 problems a run** is enough to see the checking effect over the six
  runs, and the tool's effect for llama3.2:1b on GSM8K, but not small
  differences for one model.

### What an earlier run taught

The first version of the checker took `5,000 * (1 + 0.025) = 5,000 * 1.025 =
5,125` as stating 5, checked `1/2 = 15` out of `30 bees * 1/2 = 15`, and read
a list marker on a new line as a minus sign. It told models that correct
steps were wrong. Those cases, and the others found then, are in
`tests/test_compute.py`, and the checker now checks only an operation written
out in full (see [Exact arithmetic](../docs/reference/compute.md#checking-written-arithmetic)).

Before that, a prompt that told the model to send every step to the tool
made qwen3:1.7b skip its reasoning: 108 of 200 right with the tool against
151 alone. The prompt now offers the tool for arithmetic and asks for
reasoning first.

### Run it

On GitHub: Actions, Benchmark, Run workflow; or push a branch named
`benchmark/<anything>`. Each job's summary is in its log, and its replies
are an artifact.

On your own computer, with Ollama:

```bash
curl -L -o test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
ollama pull qwen3:1.7b
python benchmarks/gsm8k.py test.jsonl --model qwen3:1.7b --problems 200 --out results.jsonl
python benchmarks/summary.py results.jsonl
```

A run that stops picks up where it left off.

## MATH-500 with small local models

Does a small model get more competition problems right when it can call
ADA for exact arithmetic and algebra? `benchmarks/math500.py` solves each
problem two ways, with the same model and settings:

- **Alone**: the model works the problem and puts its answer in \boxed{}.
- **With `ada_compute`**: the model may call ADA's tool for arithmetic,
  simplifying an expression with letters, or solving equations (`x^2 - 5x
  + 6 = 0`, `x*y = 6; x + y = 5`, `solve a*x^2 + b*x + c = 0 for x`).

| | |
|---|---|
| Data | [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), the 500 MATH test problems used to report models, checked by sha256; by default its 221 Algebra and Intermediate Algebra problems, the subjects ADA's algebra covers |
| Grading | [Math-Verify](https://github.com/huggingface/Math-Verify) 0.9.0, the grader made for MATH: it reads the boxed answer and compares it with the data's answer as mathematics, so 14/3 and \frac{14}{3} agree. It is installed for the benchmark only; ADA does not depend on it |
| Settings | Temperature 0, seed 0, thinking off, up to 3,072 tokens a reply, up to 8 rounds of tool calls |
| Test | Each condition's answers on the same problems, compared with an exact McNemar test on the problems where the two disagree |

### Run it

On GitHub: Actions, Benchmark MATH, Run workflow. Each model, condition
and shard is its own job, so the 16 jobs run side by side; the last job
puts the shards together and prints the summary, by subject and level.

On your own computer, with Ollama:

```bash
curl -L -o math500.jsonl https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl
pip install "math-verify[antlr4_13_2]==0.9.0"
ollama pull qwen3:1.7b
python benchmarks/math500.py math500.jsonl --model qwen3:1.7b --condition none --out none.jsonl
python benchmarks/math500.py math500.jsonl --model qwen3:1.7b --condition tool --out tool.jsonl
python benchmarks/math500.py --summary none.jsonl tool.jsonl
```

`--subjects` picks other subjects, `--problems` the first N of them, and
`--shard 1/4` every fourth problem from the second. A run that stops picks
up where it left off.
