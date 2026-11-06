# Related work

Papers on giving a language model exact arithmetic or a solver, on what that does for small models, and on
fully open models to test with. Each entry says what the paper shows and where it bears on these runs.
Numbers are as the papers report them; none has been reproduced here.

## A model that writes, a solver that computes

| Paper | What it shows | Where it bears here |
|---|---|---|
| PAL: Program-aided Language Models (Gao et al., 2022, [arXiv:2211.10435](https://arxiv.org/abs/2211.10435)) | The model writes a Python program as its reasoning and an interpreter runs it. Made GSM-Hard, GSM8K with larger numbers, where this beats chain of thought by a wide margin | The `program` condition is PAL's setup with this benchmark's four examples. GSM-Hard comes from this paper |
| Solving Math Word Problems by Combining Language Models With Symbolic Solvers (He-Yueya et al., 2023, [arXiv:2304.09102](https://arxiv.org/abs/2304.09102)) | The model writes variables and equations; SymPy solves them. About level with PAL on GSM8K, and about 20 points above it on harder algebra word problems | The closest to `solve`: the model writes equations, a solver does the arithmetic |
| ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving (Gou et al., 2023, [arXiv:2309.17452](https://arxiv.org/abs/2309.17452)) | Models trained on GPT-4 trajectories that interleave reasoning with calls to computation libraries and solvers; interleaving does better than programs alone or prose alone | Training on traces is the step these runs have not taken: here the models are used as they are |

## Small models

| Paper | What it shows | Where it bears here |
|---|---|---|
| TinyGSM (Liu et al., 2023, [arXiv:2312.09241](https://arxiv.org/abs/2312.09241)) | A 1.3B model fine-tuned on 12.3M GPT-3.5-written problems with Python solutions goes from 44.6% to 68.2% on GSM8K; with a 1.3B verifier choosing among its samples, 81.5% | A small model learns to formalize when trained on it, and choosing among several attempts adds more. Its data was written by GPT-3.5, a closed model |
| Agent Distillation (Kang et al., 2025, [arXiv:2505.17612](https://arxiv.org/abs/2505.17612)) | Tool-using agent behaviour distilled from a 32B teacher into 0.5B to 7B students; the 1.5B student does about as well as a 3B student trained on chain of thought, on MATH and GSM-Hard among others | Tool use can be taught to models as small as those here |
| T1: Tool-integrated Verification for Test-time Compute Scaling in Small Language Models (2025, [arXiv:2504.04718](https://arxiv.org/abs/2504.04718)) | Tool-based checks let small models such as Qwen2.5-1.5B choose among their samples better on MATH-500 and GSM8K | Close to using ADA's check, and the agreement of the model's answer with ADA's, as a verifier |
| rStar-Math (Guan et al., 2025, [arXiv:2501.04519](https://arxiv.org/abs/2501.04519)) | Small models trained with tree search and a process reward model over four rounds; Qwen2.5-Math-7B reaches 90.0% on MATH | What heavy training does for small models on competition problems; far from the setup here |

## Fully open models

| Model | What is open | Notes |
|---|---|---|
| SmolLM2, 135M to 1.7B (Allal et al., 2025, [arXiv:2502.02737](https://arxiv.org/abs/2502.02737)) | Weights and every training data set the authors made | smollm2:1.7b is one of the 13 models here; its base model gets about 31% on GSM8K (5-shot) by the paper |
| OLMo 2, from 1B (OLMo team, 2025, [arXiv:2501.00656](https://arxiv.org/abs/2501.00656); [OLMo-2-0425-1B-Instruct](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct)) | Weights, code and data under Apache 2.0 | Not yet in these runs |
| Open-1B (Donaghy et al., 2026, [arXiv:2609.17380](https://arxiv.org/abs/2609.17380)) | Weights, pretraining data, every checkpoint, code, and a training run each step of which can be reproduced bit for bit | Trained on about a tenth of OLMo 2 1B's tokens, and behind it on benchmarks by the paper |
