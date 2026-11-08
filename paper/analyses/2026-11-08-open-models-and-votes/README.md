# Fully open models, votes and ADA, tested on new problems (2026-11-08)

**Question.** How far do ADA and voting take small models whose weights, training data and training code
are all open? With five samples each, does a model that writes equations for ADA to solve (`retry`) get
more right than the same model working alone (`none`)? And how do a reply held to a JSON schema (`json`) and a
program the model writes (`program`) compare?

This page records what the test should show, **before** the run. It replaces
[2026-11-06-retry-and-programs](../2026-11-06-retry-and-programs/README.md), which was not run.

## The models

Three instruction-tuned models from 1B to 3B parameters whose makers publish the weights, every training data
set and the training code, so anyone can train them again from scratch:

| Model | Maker | Name in the run | Weights |
|---|---|---|---|
| OLMo 2 1B Instruct (April 2025) | Ai2 | `olmo2:1b-instruct` | `hf.co/allenai/OLMo-2-0425-1B-Instruct-GGUF:Q8_0`, Ai2's own GGUF |
| SmolLM2 1.7B Instruct | Hugging Face | `smollm2:1.7b` | Ollama's library (Q8_0), as in the earlier runs |
| SmolLM3 3B | Hugging Face | `smollm3:3b` | `hf.co/ggml-org/SmolLM3-3B-GGUF:Q8_0` |

SmolLM3 works through a problem at length ("thinks") unless its system prompt ends with `/no_think`. Thinking
is off here, as for every model in the GSM8K runs, so its system prompts end with `/no_think`.

Left out: SmolLM2's 135M and 360M models, since `solve` did not help the two models under 1B in the earlier
runs even on GSM-Hard (qwen2.5:0.5b 47 against 48 alone, qwen3:0.6b 35 against 81, in
[2026-11-05-gsm-ladder](../../runs/2026-11-05-gsm-ladder/README.md)); math-tuned small models with open
weights, such as `qwen2-math:1.5b` of the earlier runs, since their training data is not published; and OLMo
3, whose smallest model has 7B parameters.

## The run

| | |
|---|---|
| Problems | GSM8K and GSM-Hard problems 600 to 1318 (`--problems 1319 --skip 600`), 719 of each |
| Conditions | `none`, `check`, `solve`, `json`, `retry`, `program` and `combine` |
| Samples | `--samples 5 --sampled none,retry,program`: `none`, `retry` and `program` are each asked five times, the first at temperature 0 and seed 0, the others at temperature 0.7 with seeds 1 to 4. `check`, `solve`, `json` and `combine` are asked once; `combine` takes the first replies of `none`, `check` and `solve` |
| Settings | As in the earlier runs: 8,192 tokens of context and at most 2,048 tokens a reply. `--system-suffix /no_think` for `smollm3:3b` only |
| Code | Commit `0a845de` of `main` |
| Where | Kaggle, two kernels side by side, one per data set, each with two T4 GPUs, one Ollama server per GPU and six shards per model ([kaggle_ladder.py](../../../benchmarks/kaggle_ladder.py)). `olmo2:1b-instruct` and `smollm2:1.7b` run on one GPU, `smollm3:3b` on the other |
| Time | About 12 kernel-hours in all, about 6 for each kernel, set by the GPU with two models; estimated from the seconds replies of models of these sizes took in the earlier runs |

Before the run, a smoke test asks GSM8K and GSM-Hard problems 0 to 19, which the earlier runs already asked,
with every model, condition and sample. It checks that the models load from Hugging Face, that the replies
under `json` follow the schema and that `smollm3:3b` does not think. Its replies count in no test.

Two kernels of the plan this page replaces ran on 2026-11-06 from 04:04 to 04:16 UTC and were then deleted.
`smollm2:1.7b` was among their models. Their replies were never downloaded or read.

## The tests

Pooled over the three models, on the problems every sample of the compared conditions finished:

| | Test | Holds when |
|---|---|---|
| 1 | **Main.** With large numbers, ADA with a vote beats the model alone with a vote | On GSM-Hard, `retry`'s vote of five gets more right than `none`'s vote of five, at p < 0.05 |
| 2 | With small numbers, ADA solving does not help ([finding 3](../../findings.md)) | On GSM8K, `none`'s vote of five gets more right than `retry`'s vote of five |
| 3 | Voting helps | For `none` and for `retry`, on each data set, the vote gets more right than the first reply |
| 4 | Asking again helps `solve` ([finding 7](../../findings.md)) | On each data set, `retry` gets more right than `solve`, first reply against first reply |
| 5 | The schema helps `solve` | On each data set, `json` gets more right than `solve` |
| 6 | The combining rule holds on new problems and models ([finding 4](../../findings.md)) | On GSM-Hard, `combine` gets more right than `none`, `check` and `solve` each; on GSM8K, `combine` is not worse than `none` at p < 0.05 |

`program` against `retry`, by first reply and by vote, is reported both ways for each data set, with no
direction set beforehand: it compares ADA with the usual way of giving a model exact arithmetic. Each test is
also reported per model.

## How it will be scored

`python paper/analysis.py` on the run's folder, at commit `0a845de` or later. Tests 4 to 6 read the tables of
first replies: each condition's total over the three models, and `combine` against the model alone. Tests 1 to
3 read the vote tables' "All" rows: the column "Vote against first reply" for test 3, and the row of `retry`
against `none`, column "Vote", in the table of each condition against each one before it, for tests 1 and 2.
The rows of `program` against `retry` in that table give the comparison reported both ways. p is the exact
McNemar test on the problems where the two answers differ. The vote is the answer most samples give, the
earliest of them when answers tie; the data set's answer grades the vote and never picks the sample.
