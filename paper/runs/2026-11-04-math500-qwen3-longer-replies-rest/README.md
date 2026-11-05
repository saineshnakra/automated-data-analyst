# MATH-500, qwen3:1.7b, the rest of the first replies asked again (2026-11-04)

**Question.** Do the rest of the first replies that gave no answer in 14,336 tokens give one with 30,720,
and what do the first sample and the votes get then?

**Answer in short.** They all do. This folder is two runs: 41868380238 was cancelled after five minutes and
asked nothing, and 41870076390 asks the 25 left (the 7 that ran out of time in run 41817241401, the 16 it
never asked and the 2 problems with no reply). With run 41817241401's 16, all 41 first replies now answer,
and 18 are right. Over all 500 problems, the first sample gets 431 right, against 413 before, and the vote of
every answer 443, against 438.

## What ran

| | |
|---|---|
| Started from | The records of runs 41768840703, 41817241401 and 41868380238, oldest first ([2026-11-04-math500-qwen3-votes](../2026-11-04-math500-qwen3-votes/README.md) and [2026-11-04-math500-qwen3-longer-replies](../2026-11-04-math500-qwen3-longer-replies/README.md)); a later run's record takes the place of an earlier one's. A first reply with no answer in fewer tokens, or one that did not end in time, is asked again with the same seed, and so is the check that starts from it; every other record stands |
| Data | MATH-500's `test.jsonl` (sha256 `35dc4108…a06132`), all 500 problems in 125 shards. Answers are graded by Math-Verify 0.9.0 |
| Models | qwen3:1.7b with thinking on, Ollama's `qwen3:1.7b` |
| Conditions | `none` (alone) and `check`, which starts from the `none` reply |
| Settings | Thinking on: temperature 0.6, top-p 0.95, top-k 20, seed 0. At most 30,720 tokens a reply in 32,768 tokens of context, and a reply may take half a second a token, 15,360 seconds (4.3 hours). One sample; the votes' other samples stand |
| Where | GitHub Actions, `benchmark-math.yml` on `main`, started by hand, on `ubuntu-latest` runners, CPU only. Run [41868380238](https://github.com/saineshnakra/automated-data-analyst/actions/runs/41868380238), commit `3c13ab3`: started 16:33 UTC and cancelled at 16:38 UTC, to start again with #175 (PR #175). Run [41870076390](https://github.com/saineshnakra/automated-data-analyst/actions/runs/41870076390): one job that gathers the earlier results and 125 shard jobs, all of which succeeded; started 16:42 UTC, the last job ended 21:14 UTC |
| Code | Commit `c3c8169` for every new record |

[config.json](config.json) has these settings in full. The records in [results.jsonl.gz](results.jsonl.gz)
are math500.py's, as run 41870076390's 125 shards wrote them, with the data set added: 1,591 records, one per
problem, condition and sample.

- **1,539 are kept from earlier runs**, as they wrote them: 875 first made by run 41725437147 (with the
  fields run 41768840703 added), 628 by run 41768840703 and 36 by run 41817241401.
- **52 are this run's own**: 25 first replies and 27 checks. 48 take the place of earlier records, among them
  run 41817241401's 7 first replies that ran out of time and their checks; 4 are new, the first replies and
  checks of the 2 problems with no reply.
- **Run 41868380238 adds nothing.** Its 42 uploaded shards hold only copies of runs 41768840703's and
  41817241401's records, and every one of them is also in run 41870076390's shards, so the folder is made
  from run 41870076390's shards alone.

## Results

Made with `python paper/analysis.py paper/runs/2026-11-04-math500-qwen3-longer-replies-rest`. analysis.py
reads each problem's first sample only; the votes are below it. The p-value is an exact McNemar test against
the model alone.

### MATH-500

Problems right, on the problems every condition finished:

| Model | Problems | none | check |
|---|---|---|---|
| qwen3:1.7b | 500 | 431 | 431 |

Against the model alone: right only with the condition / right only alone, and p:

| Model | check |
|---|---|
| qwen3:1.7b | 0 / 0, p = 1 |

Wrong answers alone, and how many of them have an arithmetic slip ADA finds:

- qwen3:1.7b: 8 of 69

Replies that failed or never ended: 0

## What it shows

- **All 41 first replies that had no answer now answer, and 18 are right.** With up to 30,720 tokens, a
  reply takes 55 minutes at the median and up to 3.9 hours, 51.6 runner-hours for the 41 (these include run
  41817241401's 16).
- **On all 500 problems**, computed from the records' `agrees_with` (benchmarks/README.md gives the same
  numbers):

| | Before (2026-11-04-math500-qwen3-votes) | Now |
|---|---|---|
| First sample | 413 | 431 |
| Vote of every answer | 438 | 443 |
| Vote of the answers ADA found no wrong equation in | 438 | 443 |
| Vote of the checked answers | 440 | 444 |
| Some sample right | 447 | 454 |
| No answer in any sample | 17 | 0 |

- **The vote gains less than the first sample.** By benchmarks/README.md, against the vote before, 8
  problems are right only now and 3 only before (exact McNemar p = 0.23); on 2 of the 3, the new first
  answer is wrong and ties with a right one, and a tie goes to the answer given first.
- **ADA in the thinking.** By benchmarks/README.md, `check_arithmetic` finds a wrong equation in the thinking
  of 6 of the 18 right replies and 9 of the 23 wrong ones, so a slip in the thinking does not tell a right
  reply from a wrong one.

## What to keep in mind

- The 41 first replies from this run and run 41817241401 have up to 30,720 tokens; every other first reply
  has up to 14,336. The first sample mixes the two.
- This folder holds the earlier runs' records too: pooling it with the folders before it counts them twice.
- Reply times depend on the runner; one runner takes twice as long as another per character of thinking
  (benchmarks/README.md, on run 41725437147).

## Before and after

- Before it: [2026-11-04-math500-qwen3-longer-replies](../2026-11-04-math500-qwen3-longer-replies/README.md),
  run 41817241401, which asked the first 23.
- After it: no later MATH-500 run is archived yet. The next run archived is
  [2026-11-05-gsm-solve-pilot](../2026-11-05-gsm-solve-pilot/README.md).
- Recorded in [benchmarks/README.md](../../../benchmarks/README.md) ("More room for the first replies with
  no answer") by PR #176.
