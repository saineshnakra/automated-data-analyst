# MATH-500, qwen3:1.7b votes on the problems left open (2026-11-04)

**Question.** Does a vote over up to five samples, with or without the replies ADA finds a wrong equation in,
get more problems right than the first reply?

**Answer in short.** The vote does; ADA does not change it. Over all 500 problems, the vote of every answer
gets 438 right against 413 for the first sample (right only with the vote 27, only with the first sample 2).
Leaving out the replies ADA finds a wrong equation in changes no vote (438); the vote of the checked answers
gets 440.

## What ran

| | |
|---|---|
| Started from | The records of run 41725437147 ([2026-11-03-math500-qwen3-thinking](../2026-11-03-math500-qwen3-thinking/README.md)). Every first answer stands except where the first reply gave no answer or thought for more than 20,000 characters; there, and on the 22 problems that run did not reach, the model is asked up to five times |
| Data | MATH-500's `test.jsonl` (sha256 `35dc4108…a06132`), all 500 problems. Answers are graded by Math-Verify 0.9.0 |
| Models | qwen3:1.7b with thinking on, Ollama's `qwen3:1.7b` |
| Conditions | `none` (alone) and `check`, which starts from the `none` reply of the same sample; both vote |
| Settings | Thinking on: temperature 0.6, top-p 0.95, top-k 20, seeds 0 to 4 (the sample number), at most 14,336 tokens a reply in 16,384 tokens of context, an hour a reply. Up to five samples; asking stops as soon as more samples cannot change a vote. Two answers are the same when Math-Verify finds them equal, and a tie goes to the answer given first |
| Where | GitHub Actions, run [41768840703](https://github.com/saineshnakra/automated-data-analyst/actions/runs/41768840703) of `benchmark-math.yml` on `main`, started by hand: 60 jobs on `ubuntu-latest` runners, CPU only, one per shard; started 03:43 UTC, the last job ended 10:43 UTC. 14 jobs ran out of time, so the run reads "failure" |
| Code | Commit `886016d` |

[config.json](config.json) has these settings in full. The records in [results.jsonl.gz](results.jsonl.gz)
are math500.py's, as this run's 60 shards wrote them, with the data set added: 1,587 records, one per
problem, condition and sample.

- **952 are run 41725437147's.** The shards wrote them back with three fields added: `sample` (0),
  `agrees_with` (the sample whose answer this one gives) and `wrong_equations` (the count ADA, as it checks at
  `886016d`, finds wrong in the reply). No other field changes.
- **635 are this run's own**: first replies and checks on 20 of the 22 problems run 41725437147 did not
  reach; the check that run did not finish; 3 checks made again because ADA now reads the reply
  differently, which replace that run's; and 591 more samples (298 alone, 293 checked) on 93 problems.

## Results

Made with `python paper/analysis.py paper/runs/2026-11-04-math500-qwen3-votes`. analysis.py reads each
problem's first sample only, so its counts are the first sample's; the votes are below it. The p-value is an
exact McNemar test against the model alone.

### MATH-500

Problems right, on the problems every condition finished:

| Model | Problems | none | check |
|---|---|---|---|
| qwen3:1.7b | 498 | 413 | 415 |

Against the model alone: right only with the condition / right only alone, and p:

| Model | check |
|---|---|
| qwen3:1.7b | 2 / 0, p = 0.5 |

Wrong answers alone, and how many of them have an arithmetic slip ADA finds:

- qwen3:1.7b: 5 of 85

Replies that failed or never ended: 3

## What it shows

The votes, out of all 500 problems, with a problem with no reply counted as missed (computed from the
records' `agrees_with`; benchmarks/README.md gives the same numbers):

| | Right |
|---|---|
| First sample | 413 |
| Vote of every answer | 438 |
| Vote of the answers ADA found no wrong equation in | 438 |
| Vote of the checked answers | 440 |
| Some sample right | 447 |

- **The vote helps**: right only with it 27, only with the first sample 2 (exact McNemar p = 2e-6 by
  benchmarks/README.md). 93 problems are asked again, 44 of them five times.
- **ADA does not change the vote.** By benchmarks/README.md, ADA finds a wrong equation in 24 replies, 4 of
  them misreadings; told of it, the model turns 2 wrong answers right and 2 right answers wrong.
- **What the vote misses**: 62 problems, 17 of them with no answer in any sample.

## What to keep in mind

- 14 jobs ran out of time, so by benchmarks/README.md the votes on 23 problems stopped early, and 2 of the 22
  problems run 41725437147 did not reach have no reply at all. analysis.py leaves those 2 out (498 problems);
  the votes above count them as missed.
- By benchmarks/README.md, the new replies take 141.5 runner-hours.
- This folder holds run 41725437147's records too. Pooling it with
  [2026-11-03-math500-qwen3-thinking](../2026-11-03-math500-qwen3-thinking/README.md) counts them twice.
- The first replies on the problems that already had an answer are from run 41725437147, with thinking up to
  14,336 tokens; 37 of them had no answer there and still have none as a first sample here.

## Before and after

- Before it: [2026-11-03-math500-qwen3-thinking](../2026-11-03-math500-qwen3-thinking/README.md), one sample
  per problem.
- After it: [2026-11-04-math500-qwen3-longer-replies](../2026-11-04-math500-qwen3-longer-replies/README.md),
  which asks the first replies with no answer again with 30,720 tokens.
- Recorded in [benchmarks/README.md](../../../benchmarks/README.md) ("Votes on the problems left open") by
  PR #172.
