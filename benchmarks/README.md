# Benchmarks

## GSM8K and GSM-Hard with small local models

Does a small model get more grade-school word problems right with ADA? Each
problem is solved in up to five ways, with the same model and settings:

- **Alone**: the model works the problem and ends with "Answer: <number>".
- **With `ada_compute`**: the model may call ADA's arithmetic tool for any
  step. A call it writes out as text instead of making is made for it, as
  `ada chat` does.
- **Alone, then checked**: the model's own answer from the first condition,
  checked by `ada.check_arithmetic`. When an equation in it is wrong, the
  model is told which, with the right value, and answers once more. When
  none is, the answer stands, so this condition only changes answers that
  have a slip in them.
- **The model writes equations, ADA solves them** (`solve`): the model
  writes the problem as equations without solving it and names the quantity
  asked for, and ADA solves the equations exactly. Every step of working
  them out is ADA's. Four worked examples, written for this benchmark and
  not taken from any data set, come before the question as earlier turns of
  the chat. Names such as `eggs_sold` are given a letter each before ADA
  reads them. ADA reads equations written in LaTeX or plain text, and leaves
  out what the model works out itself: a name defined a second time, the
  result at the end of `a = 2 / 2 = 1` and arithmetic with no name in it.
  When ADA cannot solve the equations, there is no answer; the summary also
  shows ADA's answer with the model's own in its place when there is none.
- **The model alone after worked examples** (`shots`): the same four
  example questions, worked out step by step, come before the question. It
  has as many examples as `solve`, so the two show what ADA adds beyond
  seeing examples.

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
`benchmark/<anything>`. The `models`, `datasets` and `conditions` inputs
pick what runs; by default, the three models above on both datasets with
`none,tool,check`. Each job's summary is in its log, and its replies are an
artifact.

On your own computer, with Ollama:

```bash
curl -L -o test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
ollama pull qwen3:1.7b
python benchmarks/gsm8k.py test.jsonl --model qwen3:1.7b --problems 200 --out results.jsonl
python benchmarks/summary.py results.jsonl
```

`--condition none,solve` runs some of the conditions. A run that stops
picks up where it left off.

## MATH-500 with small local models

Does a small model get more competition problems right with ADA?
`benchmarks/math500.py` solves each problem under one of five conditions,
with the same model and settings:

- **Alone** (`none`): the model works the problem and puts its answer in
  \boxed{}.
- **With `ada_compute`** (`tool`): the model may call ADA's tool for
  arithmetic, simplifying an expression with letters, or solving equations
  (`x^2 - 5x + 6 = 0`, `x*y = 6; x + y = 5`, `solve a*x^2 + b*x + c = 0 for
  x`), and the system prompt says so.
- **Alone, then checked** (`check`): the model's own answer, checked by
  `ada.check_arithmetic`. When an equation in it is wrong, the model is told
  which, with the right value, and answers once more; otherwise the answer
  stands.
- **The tool's prompt only** (`prompt`) and **the tool only** (`schema`):
  the two halves of `tool`, to tell which one changes the answers.
- **The model writes equations, ADA solves them** (`solve`), as for GSM8K.
  ADA's value is graded in LaTeX, so a root stays exact (`3 \sqrt{3}`).

| | |
|---|---|
| Data | [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), the 500 MATH test problems used to report models, checked by sha256; by default its 221 Algebra and Intermediate Algebra problems, the subjects ADA's algebra covers |
| Grading | [Math-Verify](https://github.com/huggingface/Math-Verify) 0.9.0, the grader made for MATH: it reads the boxed answer and compares it with the data's answer as mathematics, so 14/3 and \frac{14}{3} agree. It is installed for the benchmark only; ADA does not depend on it |
| Settings | Temperature 0, seed 0, thinking off, up to 3,072 tokens a reply, up to 8 rounds of tool calls |
| Test | Each condition's answers on the same problems, compared with an exact McNemar test on the problems where the two disagree |

### Results

Run [41691859618](https://github.com/saineshnakra/automated-data-analyst/actions/runs/41691859618)
on commit 8f4c150, the 221 Algebra and Intermediate Algebra problems, alone
and with the tool:

| Model | Alone | With `ada_compute` | Right only with the tool | Right only alone | Exact McNemar p |
|---|---|---|---|---|---|
| qwen2.5:1.5b | 127 (57.5%) | 100 (45.2%) | 13 | 40 | 0.0003 |
| qwen3:1.7b | 161 (72.9%) | 141 (63.8%) | 13 | 33 | 0.005 |

Offering the tool makes both models worse, and not through the tool's
answers:

- qwen3:1.7b called the tool on 4 problems, and got all 4 right with and
  without it. Every one of its 33 losses and 13 gains is on a problem where
  it made no call. What changed is the prompt and the tool's definition in
  the context: its replies got shorter (a mean of 1,654 characters against
  2,185).
- qwen2.5:1.5b called it on 24 problems. Offered the tool, it stopped putting
  its answer in \boxed{} in a third of its replies (124 boxed against 197
  alone), and 18 of its 40 losses are replies with no boxed answer.

This agrees with GSM8K, where the tool was a wash or worse for small models
and checking their answers helped. `check_arithmetic` finds a wrong equation
in 21 of the 442 replies written alone, 15 of them with a wrong final
answer, so the `check` condition is the next run, with `prompt` and `schema`
to find which half of the tool does the harm.

Run [41716634251](https://github.com/saineshnakra/automated-data-analyst/actions/runs/41716634251)
on commit f2620ed, the same 221 problems under all five conditions. Each
condition is compared with the model alone on the same problems, with the
exact McNemar p:

| Model | Alone | `tool` | `prompt` | `schema` | `check` |
|---|---|---|---|---|---|
| qwen2.5:1.5b | 125 (56.6%) | 102 (p = 0.003) | 126 (p = 1) | 100 (p = 0.001) | 124 (p = 1) |
| qwen3:1.7b | 155 (70.1%) | 139 (p = 0.03) | 148 (p = 0.32) | 140 (p = 0.03) | 162 (p = 0.04) |

- **The tool's definition is what does the harm.** With the tool offered
  under the plain prompt (`schema`), both models lose as much as with the
  whole tool. With the tool's prompt and no tool (`prompt`), they are level
  with the model alone.
- **A run does not repeat exactly on another runner, even at temperature 0
  with seed 0.** qwen3:1.7b alone got 161 right in run 1 and 155 in run 2.
  The replies it gave alone inside `check`, asked for a second time in the
  same run, got 160. A difference of a few problems between two conditions
  is within this.
- **So `check` is judged against its own first reply**, the one ADA checked:
  - ADA found a wrong equation in 8 of qwen3:1.7b's replies. Told which,
    the model turned 2 wrong answers right and none wrong. Most of the gain
    over the model alone in the table is the second asking, not the check.
  - In qwen2.5:1.5b's replies ADA found one in 12. The model fixed none
    and turned one right answer wrong.
- **ADA now finds more.** It reads LaTeX (#164) and checks steps with
  letters (#166). In the same first replies it finds a wrong equation in 17
  of qwen3:1.7b's (13 with a wrong final answer) and 26 of qwen2.5:1.5b's
  (21 wrong), against 8 and 12 before.

Since #162, `check` starts from the reply the model gave alone, so the
comparison no longer includes a second asking. The next runs turn thinking
on, for all 500 problems, and vote over several samples, with and without
the replies ADA finds a wrong equation in.

### Thinking on, all 500 problems

Run [41725437147](https://github.com/saineshnakra/automated-data-analyst/actions/runs/41725437147)
on commit 907d9a3: qwen3:1.7b with thinking on, all 500 problems in 20
parts, `none,check`. Thinking uses the sampling the model's makers recommend
(temperature 0.6, top-p 0.95, top-k 20, seed 0) and up to 14,336 tokens a
reply in 16,384 tokens of context, through Ollama's `qwen3:1.7b`, whose
weights are 4-bit (Q4_K_M). Each job may run for 5 hours 25 minutes, and
three stopped before their last problems, so 478 of the 500 have an answer
from the model alone. Of the 22 left out, 9 are level 5.

The model alone gets 399 of 478 right (83.5%). Of the 79 it misses, 37 have
no answer and 42 a wrong one.

| Level | Right | No answer |
|---|---|---|
| 1 | 40 of 42 (95.2%) | 0 |
| 2 | 82 of 86 (95.3%) | 0 |
| 3 | 95 of 101 (94.1%) | 4 |
| 4 | 101 of 124 (81.5%) | 9 |
| 5 | 81 of 125 (64.8%) | 24 |

| Subject | Right | No answer |
|---|---|---|
| Algebra | 114 of 118 (96.6%) | 0 |
| Counting & Probability | 25 of 37 (67.6%) | 3 |
| Geometry | 28 of 38 (73.7%) | 4 |
| Intermediate Algebra | 70 of 94 (74.5%) | 17 |
| Number Theory | 55 of 61 (90.2%) | 2 |
| Prealgebra | 70 of 80 (87.5%) | 3 |
| Precalculus | 37 of 50 (74.0%) | 8 |

- **Thinking helps.** On the 212 Algebra and Intermediate Algebra
  problems in both runs, the model alone gets 155 right with thinking off
  (run 1) and 184 with it on: right only with thinking 32, only without 3,
  exact McNemar p = 4e-7.
- **It is below the 93.4% the model's makers report**, with replies of up
  to 32,768 tokens and the full weights. Here 34 replies stopped while still
  thinking, with no reply, and 3 did not end within the hour a reply may
  take. Even if those 37 were right, the run would get 436 of 478 (91.2%).
- **The first 200 problems to finish overstated it**: 185 of 200 (92.5%),
  against 214 of the other 278 (77.0%). A shard finishes early when its
  problems are quick, and quick problems are easier: a mean level of 3.3
  against 3.5.
- **ADA finds almost nothing to check.** With thinking on, the reply is a
  summary written after the working, and ADA reads the reply, not the
  thinking. It flags a step in 5 of 477 replies: 2 right steps it misread
  (`4 \pi (3)^2` and `\log_{10}(10^4)`, fixed in #169), a division written
  backwards (`160 \div 570 = 3` for 570 ÷ 160 rounded down), and 2 slips. One
  slip is in a value tried and dropped, in a right answer. In the other, told
  the right value, the model gives the same wrong answer. `check` gets 401
  right, 2 more than alone, but both are replies that did not end alone and
  were asked again, not ADA's doing.
- **A reply takes 584 seconds on average**, 77.5 runner-hours for the 478,
  and one runner takes twice as long as another per character of thinking.

### Votes on the problems left open

Run [41768840703](https://github.com/saineshnakra/automated-data-analyst/actions/runs/41768840703)
on commit 886016d starts from run 41725437147's results, in 60 parts. It asks
up to five times (seeds 0 to 4) on the problems whose first reply gave no
answer or thought for more than 20,000 characters, and on the 22 run
41725437147 did not reach. Every other first answer stands. 14 jobs ran out
of time, so the votes on 23 problems stopped early, and 2 problems have no
reply at all. Each count is out of all 500, and a problem with no reply
counts as missed.

| | Right |
|---|---|
| First sample | 413 (82.6%) |
| Vote of every answer | 438 (87.6%) |
| Vote of the answers ADA found no wrong equation in | 438 (87.6%) |
| Vote of the checked answers | 440 (88.0%) |
| Some sample right | 447 (89.4%) |

| Level | Problems | First sample | Vote | No answer in any sample |
|---|---|---|---|---|
| 1 | 43 | 41 | 41 | 0 |
| 2 | 90 | 86 | 86 | 0 |
| 3 | 105 | 98 | 102 | 1 |
| 4 | 128 | 103 | 114 | 2 |
| 5 | 134 | 85 | 95 | 14 |

- **The vote helps.** Right only with it 27, only with the first sample 2,
  exact McNemar p = 2e-6. 93 problems were asked again, 44 of them five
  times, and 405 first answers stood. The new replies took 141.5
  runner-hours.
- **ADA does not change the vote.** Leaving out the replies ADA finds a
  wrong equation in changes no vote. ADA found one in 24 replies, and told
  of it, the model turned 2 wrong answers right and 2 right answers wrong.
  The checked vote is 2 higher. On one problem, ADA found that
  `(-1) - (-500) + 1` is 500, not 501, and the second answer is right. On
  the other, the reply alone did not end and was asked again.
- **Of the 24 flags, 4 are misreadings**: the 2 fixed in #169, a step in
  modular arithmetic (`3^3 = 6 \mod 7`) and a value in base 5 (`= 204_5`).
  The rest are real slips.
- **What the vote misses.** Of the 62 problems it misses, 17 have no answer
  in any sample, 14 of them level 5, and 45 have a wrong one. Some sample is
  right on 9 of them.

### More room for the first replies with no answer

Runs [41817241401](https://github.com/saineshnakra/automated-data-analyst/actions/runs/41817241401),
[41868380238](https://github.com/saineshnakra/automated-data-analyst/actions/runs/41868380238)
(cancelled) and [41870076390](https://github.com/saineshnakra/automated-data-analyst/actions/runs/41870076390)
ask the first reply again with up to 30,720 tokens and the same seed, on the
41 problems where it gave no answer. Each run starts from the ones before. In
the first, 44 jobs hit GitHub's limit on requests and asked nothing (fixed in
#174), and 7 replies ran past the time a reply was allowed then, 7,681
seconds (fixed in #175, which allows 4.3 hours). The last run asks the rest.

- **All 41 now answer, and 18 are right.** A reply takes 55 minutes at the
  median and up to 3.9 hours, 51.6 runner-hours in all.
- **On all 500 problems**, with the other samples as they were:

| | Before | Now |
|---|---|---|
| First sample | 413 (82.6%) | 431 (86.2%) |
| Vote of every answer | 438 (87.6%) | 443 (88.6%) |
| Vote of the answers ADA found no wrong equation in | 438 (87.6%) | 443 (88.6%) |
| Vote of the checked answers | 440 (88.0%) | 444 (88.8%) |
| Some sample right | 447 (89.4%) | 454 (90.8%) |
| No answer in any sample | 17 | 0 |

- **The vote gains less than the first sample.** Against the vote before,
  8 problems are right only now and 3 only before (exact McNemar p = 0.23).
  On 2 of the 3, the new first answer is wrong and ties with a right one, and
  a tie goes to the answer given first.
- **ADA in the thinking.** Each new reply keeps its thinking.
  `check_arithmetic` finds a wrong equation in the thinking of 6 of the 18
  right replies and 9 of the 23 wrong ones. Most are real slips the model
  makes on the way (`585 * 117 = 68,455`, `41 - 4 = 32`) and often puts right
  later, so a slip in the thinking does not tell a right reply from a wrong
  one.

### Run it

On GitHub: Actions, Benchmark MATH, Run workflow. Each model, condition
and shard is its own job, so they run side by side; the last job puts the
shards together and prints each condition against the model alone, by
subject and level. The `conditions` input picks which to run.

On your own computer, with Ollama:

```bash
curl -L -o math500.jsonl https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl
pip install "math-verify[antlr4_13_2]==0.9.0"
ollama pull qwen3:1.7b
python benchmarks/math500.py math500.jsonl --model qwen3:1.7b --condition none --out none.jsonl
python benchmarks/math500.py math500.jsonl --model qwen3:1.7b --condition check --out check.jsonl
python benchmarks/math500.py --summary none.jsonl check.jsonl
```

`--subjects` picks other subjects, `--problems` the first N of them, and
`--shard 1/4` every fourth problem from the second. `--condition none,check`
asks the model once and checks that reply, rather than asking twice for the
same reply. `--think` turns on a thinking model's thinking (Qwen3's), with
the sampling its makers recommend (temperature 0.6, top-p 0.95, top-k 20,
seed 0, so a run repeats) and up to 14,336 tokens a reply.
`--think-tokens 30720` gives each reply more room. A reply in the results
that gave no answer in fewer tokens is asked again with the larger room,
with the same seed, and so is the check that started from it. So is a
reply that did not end in time: a thinking reply may take an hour, or half
a second a token when it may run longer. Every other reply stands. With
thinking on, each record keeps the thinking. A run that
stops picks up where it left off.

On GitHub, the `models`, `shards`, `parts` and `think` inputs pick the
models, which shards to run and into how many parts the problems are split,
and whether thinking is on (`think` set to a number turns it on with that
many tokens a reply): all 500 problems with thinking on take
`subjects` set to every subject and around 20 parts, so each job stays
under the six-hour limit.

### Votes of several samples

With thinking on, a reply depends on the seed. `--samples 5` asks up to five
times, with seeds 0 to 4, and takes the answer most replies give, in two
ways:

- the vote of every reply's answer;
- the vote of the answers of replies `check_arithmetic` finds no wrong
  equation in. When it finds one in every reply, every reply votes.

Two answers are the same when Math-Verify finds them equal, so the votes
use no gold answer. A tie goes to the answer given first. Asking stops as
soon as more replies cannot change a vote: with five samples, three equal
answers with no wrong equation settle it. So the votes are the votes of all
five replies, at less cost. With `none,check`, the checked answers vote too.

The summary compares each vote with the first sample alone, the one
`--samples 1` gives. It also counts the problems where some sample is
right, the most any vote can reach. On GitHub, the `samples` input sets the
number. The `resume` input takes an earlier run's id, or several separated
by commas, oldest first, and each shard starts from those runs' results for
its problems; a record from a later run takes the place of one from an
earlier run. One job downloads the earlier results and passes them to the
shards, so a run with many shards stays under GitHub's limit on requests. A run with five samples
can then build on a run with one, as long as it uses the same subjects and
conditions. The problems can be split into a different number of parts:
each shard takes its own problems from every shard of the earlier run.

With thinking on, each sample can take minutes, and the hardest problems
take the longest. `--resample-above 20000` (the `resample_above` input)
asks again only when the first reply gave no answer or its thinking ran
over 20,000 characters, and lets every other first answer stand. The
threshold was picked on the first 200 problems of run 41725437147 to
finish, where those replies are 25 of 200 and hold 12 of the 15 misses. On
the other 278, which are harder, they are 74 of 278 and hold 54 of the 64
misses, and the 204 first answers left to stand are 95.1% right.

When a run picks up earlier results, a `check` record whose reply ADA now
reads differently is made again, with ADA as it checks now. This happens
when `check_arithmetic` finds a slip it did not find before. The check
starts from the model's reply alone, which is in the results, so the model
is asked again only to correct a slip.
