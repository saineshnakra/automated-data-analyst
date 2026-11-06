# References

The papers closest to this work: language models that call tools, code or solvers, and small models made
better at math, reasoning and science. Every paper here was accepted to the main conference of ICML, ICLR or
NeurIPS, or to the NeurIPS Datasets and Benchmarks track, where that is said. Workshop papers, journals, other
venues and arXiv-only work are left out.

Of the 25 papers, 19 were chosen for an oral or a spotlight, the conferences' highest presentation slots. The
other 6 were presented as posters in the main conference, as most accepted papers are; they are kept because
this work takes something from each: the benchmarks (MATH, MATH-500, GSM-Hard), the `program` condition, the
closest design to `solve`, or the plan to train a small model on answers ADA confirms. Each table marks which
is which.

Each venue and presentation was checked on the paper's OpenReview page and the conference's own pages; the
links go there. [references.bib](references.bib) has a BibTeX entry for each. The PDFs are not copied here,
since most of these papers' licences do not allow it; each OpenReview page links to its PDF.

## A model calls tools

| Paper | Venue | Presentation | What it shows | Where it bears here |
|---|---|---|---|---|
| Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al.) | NeurIPS 2023 | [Oral](https://openreview.net/forum?id=Yacmpz84TH) | GPT-J (6.7B) learns, without human labels, when and how to call tools, a calculator among them | A model of 6.7B learns to hand arithmetic to a tool; the models here are smaller and are told when to |
| ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings (Hao et al.) | NeurIPS 2023 | [Oral](https://openreview.net/forum?id=BHXsb69bSx) | Each tool is a learned token, so a frozen model calls many tools, numerical reasoning among the tasks | Another way of teaching a frozen model to call tools |
| ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al.) | ICLR 2023 | [Oral](https://openreview.net/forum?id=WE_vluYUL-X) (notable top 5%) | Interleaving reasoning with actions does better than either alone | The pattern behind `ada chat`'s tool loop |
| ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs (Qin et al.) | ICLR 2024 | [Spotlight](https://openreview.net/forum?id=dHng2O0Jjr) | An open model fine-tuned on 16,000 APIs learns to use them | Tool use trained into an open model |
| Knowing When to Ask: Segment-Level Credit Assignment for LLM Tool Use (Kumar et al.) | NeurIPS 2026 | [Spotlight](https://neurips.cc/virtual/2026/poster/149224) (decision list; the OpenReview page is not public yet) | 3B and 7B models trained with reinforcement learning call a calculator or code only when needed, with fewer needless calls | Offering `ada_compute` made small models worse here; this teaches them when a call is worth it |
| To Infinity and Beyond: Tool-Use Unlocks Length Generalization in State Space Models (Malach et al.) | ICLR 2026 | [Oral](https://openreview.net/forum?id=sSfep4udCb) | Proves that state space models cannot do long-form generation on their own but can with tools; small models with tools (130M to 1.4B) solve arithmetic of any length | A reason exact arithmetic belongs in a tool, not the model |

## Code and solvers for math and science

| Paper | Venue | Presentation | What it shows | Where it bears here |
|---|---|---|---|---|
| PAL: Program-aided Language Models (Gao et al.) | ICML 2023 | [Poster](https://openreview.net/forum?id=M1fd9Z00sj) (main track) | The model writes a Python program and an interpreter runs it; made GSM-Hard | The `program` condition is PAL's setup, and GSM-Hard is its data set |
| SatLM: Satisfiability-Aided Language Models Using Declarative Prompting (Ye et al.) | NeurIPS 2023 | [Poster](https://openreview.net/forum?id=TqW5PL1Poi) (main track) | The model writes constraints and a solver (Z3) finds the answer; beats programs on arithmetic and logic | The closest design to `solve`: the model states the problem, a solver works it out |
| ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving (Gou et al.) | ICLR 2024 | [Poster](https://openreview.net/forum?id=Ep0TtjVoap) (main track) | Open models (7B to 70B) fine-tuned to interleave reasoning with programs that use libraries such as SymPy | Tool use trained into open models for math, which these runs have not done |
| MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning (Yue et al.) | ICLR 2024 | [Spotlight](https://openreview.net/forum?id=yLClGs770I) | Tuning on a mix of reasoning in words and programs run by an interpreter gives large gains across math benchmarks | Reasoning in words and programs run by a tool each help on different problems, as the model's own answer and ADA's do here |
| OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset (Toshniwal et al.) | NeurIPS 2024, Datasets and Benchmarks | [Oral](https://openreview.net/forum?id=Mbd3QxXjq5) | 1.8M code-interpreter solutions for GSM8K and MATH, written by an open model under a permissive licence | Openly licensed training data of the kind a fully replicable small model needs |
| LLM-SR: Scientific Equation Discovery via Programming with Large Language Models (Shojaee et al.) | ICLR 2025 | [Oral](https://openreview.net/forum?id=m2nmp8P5in) | The model proposes equations as programs; a numerical optimizer fits them to data | The model proposes the structure and the numbers are worked out elsewhere, as with ADA |
| LeanDojo: Theorem Proving with Retrieval-Augmented Language Models (Yang et al.) | NeurIPS 2023, Datasets and Benchmarks | [Oral](https://openreview.net/forum?id=g7OX2sOJtn) | A small prover trained in one GPU-week, every proof checked by Lean | A small model paired with an exact checker |
| Lean-STaR: Learning to Interleave Thinking and Proving (Lin et al.) | ICLR 2025 | [Spotlight](https://openreview.net/forum?id=SOWZ59UyNc) | A 7B model is fine-tuned on proofs it wrote that Lean confirms | Training on outputs an exact checker confirms, as ADA could confirm equations |

## Small models made stronger

| Paper | Venue | Presentation | What it shows | Where it bears here |
|---|---|---|---|---|
| Specializing Smaller Language Models towards Multi-Step Reasoning (Fu et al.) | ICML 2023 | [Oral](https://openreview.net/forum?id=MXuLl38AEm) | Distilling reasoning into models from 250M to 11B moves their ability towards math, at a cost elsewhere | Small models can be made specialists |
| Distilling LLM Agent into Small Models with Retrieval and Code Tools (Kang et al.) | NeurIPS 2025 | [Spotlight](https://openreview.net/forum?id=VkicTqszOn) | Tool-using behaviour distilled into 0.5B to 7B models; they match models one size larger trained without tools, on math among other tasks | The closest to the goal here: tools make small models do the work of larger ones |
| rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking (Guan et al.) | ICML 2025 | [Oral](https://openreview.net/forum?id=5zwF1GizFa) | Small models trained on reasoning whose steps Python checks, with tree search; a 7B model reaches 90.0% on MATH | Executed checks of each step, at a training scale far beyond these runs |
| Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning (Snell et al.) | ICLR 2025 | [Oral](https://openreview.net/forum?id=4FWAwZtd2n) | Spending more compute at answer time lets a small model beat one about 14 times larger, at the same compute, on problems it can partly solve | Asking several times and voting on ADA's answers is a cheap form of this |
| MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models (Yu et al.) | ICLR 2024 | [Spotlight](https://openreview.net/forum?id=N8N0hgNDRt) | Rewriting questions forward and backward makes training data that lifts 7B models on GSM8K and MATH | A way to grow training data for writing equations |
| STaR: Bootstrapping Reasoning With Reasoning (Zelikman et al.) | NeurIPS 2022 | [Poster](https://openreview.net/forum?id=_3ELRdg2sgI) (main track) | A model fine-tuned again and again on its own answers that are right improves on arithmetic and GSM8K | The plan for a fully open model: keep the equations ADA solves to the right answer and train on them |
| Training Language Models to Self-Correct via Reinforcement Learning (Kumar et al.) | ICLR 2025 | [Oral](https://openreview.net/forum?id=CjwERcAU7w) | Reinforcement learning on a model's own attempts teaches it to correct itself, by 15.6 points on MATH | What `check` and `retry` do with one more reply, learned instead of prompted |

## Agents for science and data

| Paper | Venue | Presentation | What it shows | Where it bears here |
|---|---|---|---|---|
| In-the-Flow Agentic System Optimization for Effective Planning and Tool Use (Li et al.) | ICLR 2026 | [Oral](https://openreview.net/forum?id=Mf5AleTUVK) | A 7B planner trained with reinforcement learning inside a loop of tools, a Python coder among them, gains on math and science | A small planner with tools, as ADA is meant to serve |
| MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science (Xu et al.) | ICLR 2026 | [Oral](https://openreview.net/forum?id=jHDZEUgS4r) | 72,413 data-science tasks with checkable answers; 7B and 14B models trained on them come close to gpt-4o | Data analysis with checkable answers, ADA's own ground |

## The benchmarks used here

| Paper | Venue | Presentation | What it shows | Where it bears here |
|---|---|---|---|---|
| Measuring Mathematical Problem Solving With the MATH Dataset (Hendrycks et al.) | NeurIPS 2021, Datasets and Benchmarks | [Poster](https://openreview.net/forum?id=7Bywt2mQsCe) | 12,500 competition problems with worked solutions | The MATH-500 runs use its problems |
| Let's Verify Step by Step (Lightman et al.) | ICLR 2024 | [Poster](https://openreview.net/forum?id=v8L0pN6EOi) (main track) | Feedback on each step trains better verifiers than feedback on the answer; picks the 500 test problems known as MATH-500 | The source of MATH-500 |

GSM8K's own paper (Cobbe et al., 2021, "Training Verifiers to Solve Math Word Problems") was not published at
a conference, so it is cited from arXiv in the paper but not listed here.
