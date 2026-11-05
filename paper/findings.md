# What helps small models at math, and where ADA helps

What the runs in [runs/](runs/) show so far, for models from 0.5B to 4B parameters run through Ollama, on
GSM8K, GSM-Hard (GSM8K's problems with larger numbers) and MATH-500's algebra. Each finding names the runs
it comes from; their pages have the tables, the settings and the replies. A finding marked **tested** was
chosen on problems 0 to 299 and held on problems 300 to 599, as set down beforehand in
[analyses/](analyses/).

## Findings

1. **Most of a small model's wrong answers are not arithmetic slips.** Over 13 models, ADA finds a wrong
   equation in 113 of the 1,247 wrong answers on GSM8K, 966 of the 2,529 on GSM-Hard, and 299 of the 1,726
   on MATH-500's algebra. With small numbers, nearly all the mistakes are somewhere other than the
   arithmetic.
   ([2026-11-05-gsm-ladder](runs/2026-11-05-gsm-ladder/README.md),
   [2026-11-05-math500-ladder](runs/2026-11-05-math500-ladder/README.md))

2. **Checking the model's arithmetic helps a little everywhere and costs almost nothing.** When ADA finds a
   wrong equation, the model is told which and answers once more. Pooled over 13 models, right only when
   checked / right only alone: GSM8K 22 / 0 and 28 / 0, GSM-Hard 145 / 5 and 146 / 9 (problems 0 to 299 and
   300 to 599), MATH-500's algebra 30 / 7.
   ([2026-11-05-gsm-ladder](runs/2026-11-05-gsm-ladder/README.md),
   [2026-11-05-gsm-combine](runs/2026-11-05-gsm-combine/README.md),
   [2026-11-05-math500-ladder](runs/2026-11-05-math500-ladder/README.md))

3. **Letting ADA do all the arithmetic helps only when the numbers are large.** Under `solve` the model
   writes the problem as equations and ADA solves them. On GSM-Hard this gets 1,597 right against 1,371
   alone on problems 0 to 299, and 1,529 against 1,405 on problems 300 to 599, at p < 0.01 for six models
   from llama3.2:1b up. On GSM8K it is worse for every model (1,650 against 2,653), and on MATH-500's
   algebra worse for every model and at every level (551 against 2,213). With ADA solving, the size of the
   numbers stops mattering much: llama3.2:3b gets 200 on GSM8K and 195 on GSM-Hard under `solve`, against
   242 and 86 alone. The smallest models hardly write equations: gemma3:1b gives a number of its own on 175
   of 300 GSM8K problems.
   ([2026-11-05-gsm-ladder](runs/2026-11-05-gsm-ladder/README.md),
   [2026-11-05-gsm-combine](runs/2026-11-05-gsm-combine/README.md),
   [2026-11-05-math500-ladder](runs/2026-11-05-math500-ladder/README.md))

4. **Tested: a fixed rule that combines the two answers gets most of both.** It takes the answer both give
   when they agree, the checked one when ADA has none, and when they differ ADA's if ADA found a wrong
   equation in the model's own answer or the question has a number of five or more digits, the model's
   otherwise. It asks for no more replies than `none`, `check` and `solve` already made. On problems 300 to
   599, GSM-Hard: 1,924 of 3,900 right, against 1,405 alone, 1,542 checked and 1,529 under `solve`. GSM8K:
   2,618, against 2,606 alone, but 16 short of `check`'s 2,634. When the two answers agree, they are right
   on 94% of GSM8K problems and 89% of GSM-Hard ones.
   ([analyses/2026-11-05-combine-rules](analyses/2026-11-05-combine-rules/README.md),
   [2026-11-05-gsm-combine](runs/2026-11-05-gsm-combine/README.md))

5. **Letting the model choose between its answer and ADA's is worse than the rule.** Shown ADA's equations
   and value beside its own answer, the model gets 2,502 right on GSM8K (137 right only after choosing, 241
   right only alone) and 1,574 on GSM-Hard, against the rule's 2,618 and 1,924.
   ([2026-11-05-gsm-combine](runs/2026-11-05-gsm-combine/README.md))

6. **Offering ADA as a tool makes small models worse, and not through the tool's answers.** With
   `ada_compute` offered, llama3.2:1b gets 23 of 200 GSM8K problems right against 85 alone, and on
   MATH-500's Algebra and Intermediate Algebra qwen2.5:1.5b gets 100 against 127 and qwen3:1.7b 141 against
   161. qwen3:1.7b calls the tool on 4 problems; what costs answers is the tool's definition in the context.
   ([2026-11-01-gsm-three-models-tool](runs/2026-11-01-gsm-three-models-tool/README.md),
   [2026-11-03-math500-tool](runs/2026-11-03-math500-tool/README.md),
   [2026-11-03-math500-five-conditions](runs/2026-11-03-math500-five-conditions/README.md))

7. **Tested: telling the model why ADA found no value, and asking once more, improves `solve` a little.**
   Over 7 models from 1.5B to 4B, `retry` beats `solve` by 30 to 52 answers in 2,100 on each data set and
   each set of problems; of the problems it asks again after the same first reply `solve` got, 14% to 24%
   come out right. Working the problem out in words first (`plan`) helps the two qwen3 models and hurts
   llama3.2:3b and qwen2.5:3b, on both sets of problems; gemma3:4b gains and phi4-mini loses on both too,
   though the tests did not name them. Neither changes the combining rule's total by more than 12 answers.
   ([analyses/2026-11-05-plan-and-retry](analyses/2026-11-05-plan-and-retry/README.md),
   [2026-11-05-gsm-plan](runs/2026-11-05-gsm-plan/README.md),
   [2026-11-05-gsm-plan-check](runs/2026-11-05-gsm-plan-check/README.md))

8. **On competition algebra, what helps most is letting the model think.** With thinking on, qwen3:1.7b
   gets 254 of 292 MATH-500 Algebra, Intermediate Algebra and Prealgebra problems right, against 207 with
   it off on the same problems (50 / 3, p = 6e-12), with up to 14,336 tokens a reply against 3,072 and its
   makers' sampling (temperature 0.6). Over all 500 problems, a vote of several samples gets 438 against 413
   for the first. ADA adds nothing there: it finds a wrong step in 5 of 477 replies, and leaving those out
   of the vote changes no answer.
   ([2026-11-03-math500-qwen3-thinking](runs/2026-11-03-math500-qwen3-thinking/README.md),
   [2026-11-04-math500-qwen3-votes](runs/2026-11-04-math500-qwen3-votes/README.md),
   [2026-11-05-math500-ladder](runs/2026-11-05-math500-ladder/README.md))

9. **Worked examples of grade-school problems cost answers.** The four examples `solve` uses, worked out
   step by step before the question (`shots`), are worse than alone: 305 / 495 on GSM8K, 252 / 335 on
   GSM-Hard and 273 / 484 on MATH-500's algebra.
   ([2026-11-05-gsm-ladder](runs/2026-11-05-gsm-ladder/README.md),
   [2026-11-05-math500-ladder](runs/2026-11-05-math500-ladder/README.md))

## Where ADA helps, in short

| Use | Where it helps | Where it doesn't |
|---|---|---|
| Check the model's arithmetic | Everywhere a little, most with large numbers | It loses at most 9 answers in 3,900 |
| The model writes equations, ADA solves them | Large numbers, models from about 1B up | Small numbers; competition algebra |
| The combining rule | Large numbers, with little cost on small ones | Competition algebra: at most 75 of 3,939 problems to gain |
| ADA offered as a tool | Not found for these models | Every run that tried it |

## What to keep in mind

- One reply per problem and condition at temperature 0 and seed 0, unless a run says otherwise. On Kaggle,
  two replies to the same message differ on about a third of GSM8K problems, about as often one way as the
  other. The pooled differences above are far beyond that; many per-model ones are not.
- Thinking is off except in finding 8, which compares one model under other settings as well.
- GSM8K is graded exactly, GSM-Hard to within 0.001 and MATH-500 by Math-Verify 0.9.0.
- The earliest runs used earlier code, such as a checker that flagged right steps as wrong; their pages say
  what changed.
