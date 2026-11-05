# MATH-500, qwen3:1.7b with thinking on, all 500 problems (2026-11-03)

**Question.** How many MATH-500 problems does qwen3:1.7b get right with thinking on, and does ADA's check
add anything?

**Answer in short.** It gets 399 of the 478 problems it reached right alone (83.5%); of the 79 it misses, 37
have no answer. ADA's check changes almost nothing: `check` gets 401, and by benchmarks/README.md both gains
are replies that did not end alone and were asked again, not ADA's doing. With thinking on, ADA reads the
summary written after the thinking, and it flags a step in 5 of the 477 checked replies.

## What ran

| | |
|---|---|
| Data | MATH-500's `test.jsonl` (sha256 `35dc4108…a06132`), all 500 problems in all seven subjects. Three jobs stopped at the job time limit before their last problems, so 478 problems have a reply alone and 477 a check. Answers are graded by Math-Verify 0.9.0 |
| Models | qwen3:1.7b with thinking on, Ollama's `qwen3:1.7b` (4-bit weights, Q4_K_M, by benchmarks/README.md) |
| Conditions | `none` (alone) and `check`, which starts from the `none` reply (#162): ADA checks every equation in it, and when one is wrong the model is told which and answers once more |
| Settings | Thinking on, with the sampling the model's makers recommend: temperature 0.6, top-p 0.95, top-k 20, seed 0. At most 14,336 tokens a reply in 16,384 tokens of context, and a reply may take an hour. One reply per problem and condition |
| Where | GitHub Actions, run [41725437147](https://github.com/saineshnakra/automated-data-analyst/actions/runs/41725437147) of `benchmark-math.yml` on `main`, started by hand: 20 jobs on `ubuntu-latest` runners, CPU only, one per shard of 25 problems; started 21:50 UTC on 2026-11-03, the last job ended 03:32 UTC on 2026-11-04. 3 jobs were cancelled at the job time limit, so the run reads "cancelled" |
| Code | Commit `907d9a3` |

[config.json](config.json) has these settings in full. The records in [results.jsonl.gz](results.jsonl.gz)
are math500.py's, one per problem and condition, with the data set added: 955 records (478 alone, 477
checked), with the model's reply, the length of its thinking (`thinking_chars`) and ADA's checks. The
thinking itself is not kept in this run.

## Results

Made with `python paper/analysis.py paper/runs/2026-11-03-math500-qwen3-thinking`. The p-value is an exact
McNemar test against the model alone.

### MATH-500

Problems right, on the problems every condition finished:

| Model | Problems | none | check |
|---|---|---|---|
| qwen3:1.7b | 477 | 399 | 401 |

Against the model alone: right only with the condition / right only alone, and p:

| Model | check |
|---|---|
| qwen3:1.7b | 2 / 0, p = 0.5 |

Wrong answers alone, and how many of them have an arithmetic slip ADA finds:

- qwen3:1.7b: 1 of 78

Replies that failed or never ended: 3

## What it shows

- **The model alone gets 399 of 478 right.** Of the 79 misses, 37 have no answer and 42 a wrong one.
  benchmarks/README.md breaks this down by level and subject: 40 of 42 right at level 1, 81 of 125 at level 5.
- **Thinking helps.** By benchmarks/README.md, on the 212 Algebra and Intermediate Algebra problems in both
  runs, the model alone gets 155 right with thinking off (run 41691859618) and 184 with it on: right only
  with thinking 32, only without 3, exact McNemar p = 4e-7.
- **ADA finds almost nothing to check.** It reads the reply, not the thinking, and the reply is a summary of
  the working. It flags a step in 5 of the 477 replies it checks; benchmarks/README.md reads all five: 2 right
  steps misread (fixed in #169), one division written backwards and 2 real slips.
- **A reply takes 584 seconds on average**, 77.5 runner-hours for the 478 replies alone.

## What to keep in mind

- The 22 problems no job reached are not a random sample: by benchmarks/README.md, 9 of them are level 5,
  and the first 200 problems to finish got 185 right (92.5%) against 214 of the other 278 (77.0%).
- 3 replies alone did not end within the hour (`test/geometry/183.json`,
  `test/intermediate_algebra/894.json`, `test/precalculus/801.json`) and count as wrong. By
  benchmarks/README.md, another 34 stopped while still thinking, with no reply.
- Temperature 0.6: the reply depends on the seed, and this run asks once per problem.
- analysis.py counts 477 problems, the ones with both conditions; the one problem without a check is a miss
  alone, so alone is 399 of 478 and 399 of 477.

## Before and after

- Before it: [2026-11-03-math500-five-conditions](../2026-11-03-math500-five-conditions/README.md), two
  models with thinking off on the algebra problems.
- After it: [2026-11-04-math500-qwen3-votes](../2026-11-04-math500-qwen3-votes/README.md), which starts from
  these records and votes over more samples on the problems left open.
- Recorded in [benchmarks/README.md](../../../benchmarks/README.md) ("Thinking on, all 500 problems") by
  PR #170.
