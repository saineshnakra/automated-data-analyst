# MATH-500, qwen3:1.7b first replies asked again with 30,720 tokens (2026-11-04)

**Question.** Do the first replies that gave no answer in 14,336 tokens give one with 30,720?

**Answer in short.** Some do. 41 problems had a first reply with no answer or no reply at all. This run asks 23
of them again: 16 now answer, 7 of those right, and 7 stop at the time limit a reply had then (7,680
seconds). 44 of its 125 jobs hit GitHub's limit on requests and asked nothing, so the other 18 wait for the
next run, [2026-11-04-math500-qwen3-longer-replies-rest](../2026-11-04-math500-qwen3-longer-replies-rest/README.md).

## What ran

| | |
|---|---|
| Started from | The records of run 41768840703 ([2026-11-04-math500-qwen3-votes](../2026-11-04-math500-qwen3-votes/README.md)), which hold run 41725437147's too. A first reply with no answer in fewer tokens than this run allows is asked again with the same seed, and so is the check that starts from it; every other record stands |
| Data | MATH-500's `test.jsonl` (sha256 `35dc4108…a06132`), all 500 problems in 125 shards; the 81 shards that ran hold 324 problems. Answers are graded by Math-Verify 0.9.0 |
| Models | qwen3:1.7b with thinking on, Ollama's `qwen3:1.7b` |
| Conditions | `none` (alone) and `check`, which starts from the `none` reply |
| Settings | Thinking on: temperature 0.6, top-p 0.95, top-k 20, seed 0. At most 30,720 tokens a reply in 32,768 tokens of context, and a reply may take a quarter of a second a token, 7,680 seconds. One sample; the votes' other samples stand |
| Where | GitHub Actions, run [41817241401](https://github.com/saineshnakra/automated-data-analyst/actions/runs/41817241401) of `benchmark-math.yml` on `main`, started by hand: 125 jobs on `ubuntu-latest` runners, CPU only, one per shard; started 10:53 UTC, the last job ended 15:13 UTC. 44 jobs failed while downloading the earlier results, with GitHub's API rate limit (PR #174), and asked and uploaded nothing, so the run reads "failure" |
| Code | Commit `076a190` |

[config.json](config.json) has these settings in full. The records in [results.jsonl.gz](results.jsonl.gz)
are math500.py's, as the 81 shards that ran wrote them, with the data set added: 992 records, one per
problem, condition and sample.

- **942 are kept from run 41768840703**, as it wrote them: 584 first made by run 41725437147 (with the
  fields run 41768840703 added) and 358 by run 41768840703.
- **50 are this run's own**, each in place of an earlier record: 23 first replies and 27 checks. 23 checks
  start from the new replies; of the other 4, 2 are checks that gave no answer in fewer tokens and 2 are made
  again because ADA now reads the reply differently. Each new reply that ended keeps its thinking
  (`thinking`).

## Results

Made with `python paper/analysis.py paper/runs/2026-11-04-math500-qwen3-longer-replies`. The tables cover the
324 problems of the 81 shards that ran, not all 500. The p-value is an exact McNemar test against the model
alone.

### MATH-500

Problems right, on the problems every condition finished:

| Model | Problems | none | check |
|---|---|---|---|
| qwen3:1.7b | 324 | 278 | 280 |

Against the model alone: right only with the condition / right only alone, and p:

| Model | check |
|---|---|
| qwen3:1.7b | 2 / 0, p = 0.5 |

Wrong answers alone, and how many of them have an arithmetic slip ADA finds:

- qwen3:1.7b: 3 of 46

Replies that failed or never ended: 9

## What it shows

- **Of the 23 first replies asked again, 16 now answer and 7 are right.** The other 7 stop at 7,680.6 to
  7,680.7 seconds, the time limit, with no answer; they are the 7 failed first replies in the table above,
  and 2 checks failed the same way.
- **The first replies with more room are slow**: the 23 new first replies take 30.0 runner-hours, and the
  run's 50 new records 43.8.

## What to keep in mind

- 18 of the 41 problems are not asked here: 16 first replies with no answer and the 2 problems with no reply,
  all in shards that failed. PR #174 says 22; the records show 18.
- Whether a reply ends within 7,680 seconds depends on the runner's speed as well as on the model (PR #175).
  The next run allows twice as long and asks the 7 again.
- This folder holds the earlier runs' records too: pooling it with the folders before it counts them twice.
  The state after all the asking again is in
  [2026-11-04-math500-qwen3-longer-replies-rest](../2026-11-04-math500-qwen3-longer-replies-rest/README.md).

## Before and after

- Before it: [2026-11-04-math500-qwen3-votes](../2026-11-04-math500-qwen3-votes/README.md).
- After it: [2026-11-04-math500-qwen3-longer-replies-rest](../2026-11-04-math500-qwen3-longer-replies-rest/README.md),
  which asks the other 18 and the 7 that ran out of time.
- Recorded in [benchmarks/README.md](../../../benchmarks/README.md) ("More room for the first replies with
  no answer") by PR #176; its two failures are described in PR #174 (the request limit) and PR #175 (the time
  limit).
