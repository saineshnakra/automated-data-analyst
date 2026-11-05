# Choosing how to combine the model's answer with ADA's (2026-11-05)

**Question.** The model's own answer and ADA's answer from `solve` are each better in a different place:
the model's on GSM8K, ADA's on GSM-Hard. Can a rule that looks only at what a run has (the two answers,
ADA's check of the model's working and the question) do as well as the better of the two everywhere?

This page records how the rule was chosen, **before** it was tested on problems it was not chosen on.

## The rules

Every rule takes the answer both give when the model's own answer (`none`) and ADA's (`solve`) agree, and
`check`'s when ADA has no answer. They differ only when the two answers differ:

| Rule | When the answers differ, take |
|---|---|
| ADA's | ADA's answer |
| check decides | ADA's when ADA found a wrong equation in the model's own answer, the model's otherwise |
| check or large numbers | as "check decides", and ADA's too when the question has a number of five or more digits |
| either right | whichever is right: the most any rule could get, not a rule |

"Five or more digits" was the best of 3, 4, 5 and 6 digits on these problems (GSM8K and GSM-Hard pooled:
3 digits 2,542 and 1,952 right; 4 digits 2,637 and 1,963; 5 digits 2,675 and 1,977; 6 digits 2,681 and
1,975). Other cues were tried and dropped because they read the data set rather than the problem: taking
ADA's answer when it is a whole number helps on GSM8K, whose answers are whole numbers, and hurts on
GSM-Hard.

## Scored on the problems the rule was chosen on

From [2026-11-05-gsm-ladder](../../runs/2026-11-05-gsm-ladder/README.md), with
`python paper/combine_rules.py paper/runs/2026-11-05-gsm-ladder --gsm8k test.jsonl --gsmhard gsmhardv2.jsonl`.
"Disagree" counts the problems where ADA had an answer and it differed from the model's.

### GSM8K, problems 0 to 299

| Model | Problems | Disagree | none | check | solve | ADA's | check decides | check or large numbers | either right |
|---|---|---|---|---|---|---|---|---|---|
| gemma3:1b | 300 | 128 | 133 | 133 | 71 | 113 | 135 | 135 | 158 |
| gemma3:4b | 300 | 89 | 256 | 256 | 149 | 202 | 256 | 256 | 262 |
| llama3.2:1b | 300 | 192 | 151 | 156 | 76 | 109 | 160 | 159 | 180 |
| llama3.2:3b | 300 | 103 | 242 | 243 | 200 | 211 | 245 | 245 | 264 |
| phi4-mini | 300 | 57 | 265 | 268 | 180 | 236 | 266 | 266 | 274 |
| qwen2-math:1.5b | 300 | 56 | 248 | 249 | 122 | 220 | 248 | 248 | 255 |
| qwen2.5:0.5b | 300 | 163 | 120 | 121 | 48 | 85 | 122 | 122 | 133 |
| qwen2.5:1.5b | 300 | 136 | 186 | 187 | 133 | 157 | 188 | 186 | 211 |
| qwen2.5:3b | 300 | 84 | 256 | 256 | 193 | 226 | 257 | 257 | 272 |
| qwen3:0.6b | 300 | 135 | 143 | 143 | 36 | 93 | 145 | 145 | 150 |
| qwen3:1.7b | 300 | 128 | 240 | 245 | 125 | 162 | 241 | 240 | 248 |
| qwen3:4b-instruct | 300 | 45 | 282 | 282 | 235 | 257 | 281 | 280 | 289 |
| smollm2:1.7b | 300 | 148 | 131 | 136 | 82 | 121 | 137 | 136 | 159 |
| All | | | 2653 | 2675 | 1650 | 2192 | 2681 | 2675 | 2855 |

### GSM-Hard, problems 0 to 299

| Model | Problems | Disagree | none | check | solve | ADA's | check decides | check or large numbers | either right |
|---|---|---|---|---|---|---|---|---|---|
| gemma3:1b | 300 | 153 | 64 | 67 | 31 | 53 | 68 | 55 | 77 |
| gemma3:4b | 300 | 138 | 152 | 165 | 149 | 176 | 179 | 186 | 209 |
| llama3.2:1b | 300 | 245 | 43 | 51 | 85 | 94 | 88 | 103 | 113 |
| llama3.2:3b | 300 | 213 | 86 | 110 | 195 | 203 | 153 | 206 | 216 |
| phi4-mini | 300 | 144 | 123 | 137 | 184 | 210 | 167 | 213 | 216 |
| qwen2-math:1.5b | 300 | 109 | 156 | 162 | 107 | 174 | 168 | 180 | 200 |
| qwen2.5:0.5b | 300 | 202 | 48 | 51 | 47 | 62 | 65 | 74 | 82 |
| qwen2.5:1.5b | 300 | 188 | 99 | 111 | 134 | 150 | 128 | 158 | 176 |
| qwen2.5:3b | 300 | 173 | 127 | 149 | 190 | 203 | 157 | 209 | 220 |
| qwen3:0.6b | 300 | 145 | 81 | 88 | 35 | 78 | 91 | 85 | 100 |
| qwen3:1.7b | 300 | 178 | 148 | 162 | 138 | 154 | 171 | 170 | 200 |
| qwen3:4b-instruct | 300 | 111 | 190 | 190 | 232 | 243 | 199 | 245 | 258 |
| smollm2:1.7b | 300 | 200 | 54 | 68 | 70 | 91 | 76 | 93 | 104 |
| All | | | 1371 | 1511 | 1597 | 1891 | 1710 | 1977 | 2171 |


"check or large numbers" is the rule kept. Pooled over the 13 models it gets 2,675 of 3,900 on GSM8K,
the same as `check` and 22 more than the model alone, and 1,977 on GSM-Hard, 86 more than taking ADA's
answer every time and 466 more than `check`. It is [benchmarks/gsm8k.py](../../../benchmarks/gsm8k.py)'s
`combine` condition. These numbers are on the problems the rule was chosen on, so they overstate it.

## The test, set before it ran

[2026-11-05-gsm-combine](../../runs/2026-11-05-gsm-combine/README.md): the same 13 models on GSM8K's and
GSM-Hard's problems 300 to 599, which no rule was chosen on, with `none`, `check`, `solve`, `combine` and
`judge`. `judge` is the other way to settle a difference: the model is shown ADA's equations and value
beside its own answer and decides. Kaggle kernel `saineshnakra/ada-benchmark-combine`, version 1, commit
`f5fe3de`.

What would count as the rule holding up, pooled over the 13 models:

- on GSM8K, `combine` gets at least as many right as `check`, and more than the model alone;
- on GSM-Hard, `combine` gets more right than `solve` and than `check`;
- per model, `combine` is not worse than the model alone at p < 0.05 on either data set.

`judge` is compared with `combine` on the same problems; there is no threshold set for it.

## How it came out

On problems 300 to 599 ([2026-11-05-gsm-combine](../../runs/2026-11-05-gsm-combine/README.md)), three of
the four tests held. On GSM-Hard `combine` got 1,924 of 3,900 right, against 1,529 for `solve` and 1,542 for
`check`, and no model did worse than alone at p < 0.05 on either data set. On GSM8K it beat the model alone
(2,618 against 2,606) but not `check` (2,634): its large-numbers branch is wrong more often than the model on
GSM8K's few questions with five-digit numbers. `judge` got 2,502 and 1,574.
