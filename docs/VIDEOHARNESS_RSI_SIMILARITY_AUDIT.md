# VideoHarness-RSI similarity and novelty audit

Audit date: 2026-11-06. Upstream snapshot: the `main` branch of [Tencent/VideoHarness-RSI](https://github.com/Tencent/VideoHarness-RSI), downloaded read-only on the audit date. Local target: `src/videogen_rsi/evolution_harness.py` and its CLI/proposer adapters.

## Code-level result

No upstream source file was copied into this project. A token-sequence comparison between the 373-line local core and every Python file in the upstream snapshot found:

- closest upstream outer-loop file, `vl_harness/meta_harness.py`: 1.37% local-token coverage in matching blocks of at least eight tokens; whole-sequence ratio 11.90%;
- closest file by long-block coverage, `vl_harness/inner_loop.py`: 2.49% local-token coverage; whole-sequence ratio 8.91%;
- 3 of 273 meaningful local lines occur exactly anywhere in upstream Python: `from __future__ import annotations`, `from pathlib import Path`, and `run_dir.mkdir(parents=True, exist_ok=True)`;
- there are no shared domain-specific class or function names between the local core and upstream `meta_harness.py`.

These numbers are engineering diagnostics rather than a legal plagiarism opinion. They support the narrower statement that the implementation is independent and has very little literal source overlap.

## Concept-level result

The outer research pattern is intentionally inherited and must be attributed:

| Axis | VideoHarness-RSI | VideoGen-RSI v1 |
|---|---|---|
| Frozen foundation model | Long-video understanding VLM | Wan video generator, separately frozen evaluator |
| Evolved object | Python context/memory harness | Generation controller artifact: diagnose, repair, generate and select |
| Unit of evaluation | Mostly deterministic QA accuracy | Stochastic generated videos paired by task, seed, model scale and budget |
| Search gate | Strict development accuracy improvement | Strict development metric improvement plus identity/naturalness guards |
| Post-search protocol | Freeze and evaluate held-out | Separate blinded calibration and admission, clustered bootstrap gate, then locked test |
| Evidence | Selected frames and answers | Generated videos, all-frame temporal evidence, event clauses and evaluator receipts |
| Cost regime | Read-time visual-token/frame budget | Expensive asynchronous GPU generation calls plus judge cost |
| Primary target | Long-video comprehension | Physical behavior and temporal correctness in video generation |

The shared high-level pattern is still substantial: a coding agent modifies a harness around a frozen model, candidates move a development frontier, and the test set stays hidden. Therefore a paper whose contribution is only “apply VideoHarness-RSI to generation” has a real incremental-novelty risk even though its code is independent.

## What must become the contribution

The defensible contribution should be a generation-specific optimization and evaluation protocol, implemented and ablated rather than claimed from scaffolding:

1. **Paired uplift frontier.** Every development unit should compare baseline and agentic generation under the same task, seed, model scale and generation-call budget. Promotion should require a positive task-cluster bootstrap lower bound at both Wan 1.3B and 14B, rather than only a higher aggregate mean.
2. **Budget-safe controller search space.** Candidates should compose typed video-generation operations such as temporal diagnosis, prompt repair, source-video restart and selection. The evaluator must reject candidates that change model weights, generation calls, seeds, benchmark requirements or judge policy.
3. **Independent physical evidence.** Search-time GPT feedback and final VLM evaluation must be separate. Final scoring should bind physical event clauses to timestamps/all-frame evidence and retain judge identity and raw receipts.
4. **Transfer claim.** The learned controller should improve unseen physical mechanisms or prompt families, at both model scales, and survive the frozen PhyGenBench protocol. A development-only gain is not RSI evidence.
5. **Ablations against the actual inheritance.** Compare fixed hand-written controller, random/search-without-experience, agentic proposal without admission, and the full harness. This identifies what is gained beyond the upstream propose/evaluate/promote template.

Until the real Wan evaluator is bound and these experiments pass, describe the local artifact as an independently implemented, VideoHarness-RSI-inspired experimental harness. Do not describe the outer-loop idea itself as novel.

## Recommended attribution language

> We adopt the frozen-model, coding-agent harness-search paradigm introduced by VideoHarness-RSI, but redesign the candidate interface and evaluation protocol for stochastic video generation. Candidates operate on generation-time diagnosis, repair and selection under matched generation budgets. Promotion and finalization use paired task/seed comparisons, independent physical-evidence judging, cross-scale guards and held-out transfer splits.

This language should be updated if the paired-uplift frontier and real evaluator are not implemented in the final system.
