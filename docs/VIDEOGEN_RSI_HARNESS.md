# VideoGen-RSI harness

This is the unified outer loop that was missing from the project. It adapts the strict development-frontier and held-out-finalization pattern of Tencent's [VideoHarness-RSI](https://github.com/Tencent/VideoHarness-RSI) to video generation. The upstream project evolves context-construction programs around a frozen video VLM. This project evolves controller artifacts around frozen Wan generators and a frozen evaluator protocol. No upstream source file is copied; the architecture and attribution are recorded here. VideoHarness-RSI is Apache-2.0 and its outer loop derives from the MIT Meta-Harness project.

The harness owns five transitions:

1. `init`: freeze commands, split manifests, promotion gates and a G0 seed.
2. `evolve`: give a coding-agent proposer only the current frontier plus development failures; seal the proposed artifact; evaluate it; promote only a strict primary gain with no configured quality regression.
3. `finalize`: run calibration and admission as complete blinded paired comparisons. Both phases must have a positive task-cluster bootstrap lower bound and no more than 25% regressions.
4. `test`: unlock the held-out split only for the exact finalized artifact.
5. `status` / `doctor`: expose durable state, command readiness, artifact hashes and the append-only hash-chained ledger.

The generator/evaluator commands live in `frozen_config.json` inside each run. A candidate cannot modify them. Candidate implementation files are sealed by SHA-256. Missing/duplicate evaluation units, unpaired admission rows, changed candidates and premature test access fail closed. Interrupted calls retain attempt receipts and can be resumed without silently retrying completed evaluations.

## Offline lifecycle check

The shipped smoke config makes no model, API or GPU calls:

```bash
PYTHONPATH=src python -m videogen_rsi.harness_cli init \
  --config configs/videogen_rsi_harness_smoke_v1.json \
  --run runs/harness_smoke_v1
PYTHONPATH=src python -m videogen_rsi.harness_cli evolve --run runs/harness_smoke_v1 --iterations 2
PYTHONPATH=src python -m videogen_rsi.harness_cli finalize --run runs/harness_smoke_v1
PYTHONPATH=src python -m videogen_rsi.harness_cli test --run runs/harness_smoke_v1
PYTHONPATH=src python -m videogen_rsi.harness_cli status --run runs/harness_smoke_v1
```

`scripts/mock_videogen_harness_component.py` exists only to verify lifecycle and locks. Its scores are synthetic and never research evidence.

## Real proposer and evaluator

`scripts/codex_videogen_harness_proposer.py` is the default coding-agent proposer adapter. It calls `codex exec` headlessly with a strict JSON schema and writes only into the candidate directory. It inherits the local Codex model unless `VIDEOGEN_HARNESS_PROPOSER_MODEL` or `--model` is set. This follows the official OpenAI pattern of structured, noninteractive Codex repair loops. Replace the smoke proposer command with:

```json
["{python}", "scripts/codex_videogen_harness_proposer.py", "--request", "{request}", "--output", "{output}", "--candidate-dir", "{candidate_dir}"]
```

The real evaluator command must be a frozen adapter that consumes `{candidate}`, `{split}`, `{split_manifest}` and `{output}`. It must retain all task/seed units, report aggregates, and for calibration/admission include `with_score`, `without_score`, `blinded`, `evaluator_role=independent`, and a versioned `evaluator_id`. The existing project components intended for that adapter are:

- generation and recovery: `generator.py`, `video_restart.py`, `gpu_ownership.py`, durable run/checkpoint scripts;
- evidence and judging: timestamped all-frame exports, `physical_clause_judge.py`, `event_judge.py`, AIHub adapters;
- repair/controller candidates: trajectory, temporal-conditioning, event-controller and source-appearance modules;
- controlled comparisons: raw-single, equal-budget raw-best2, fixed compiler and feedback arms;
- experience: `experience_cli.py`, `admission_review.py`, `memory.py`;
- final evaluation: `phygen_two_scale.py` and the frozen PhyGenBench 40→120 manifests.

The powerful VLM endpoint is deliberately outside the candidate artifact. Until that independent evaluator is qualified and bound into a production config, this harness can run development orchestration but must not unlock formal benchmark claims.

## Fixed physics evaluator

`videogen_rsi.physics_evaluator` adds a fixed, candidate-inaccessible physical boundary layer. It consumes externally extracted full-frame tracks and supports:

- gravity direction, acceleration ranges and non-event velocity continuity;
- collision momentum and restitution;
- contact, penetration and unsupported-hover distances;
- contact-conditioned friction, sliding and stopping;
- rigid shape invariants, soft-body deformation/recovery and liquid volume/fragmentation limits;
- ordered events and absolute event time windows;
- full-frame track coverage;
- reference-assisted rubrics for all 160 PhyGenBench prompts;
- joint numeric/VLM checks in which a hard failure cannot be overridden by the VLM.

Missing trajectories, masses, event anchors, signed distances or material metrics yield `unknown`; they never become passes. The evaluator does not claim to recover accurate geometry from pixels. A tracker or measurement adapter must produce the declared trace, and its provenance belongs in the frozen evaluation artifact.

```bash
PYTHONPATH=src python -m videogen_rsi.physics_cli phygen-rubric \
  --catalog data/phygenbench/prompt-rubrics.json \
  --id phygen_000000 --output rubric.json

PYTHONPATH=src python -m videogen_rsi.physics_cli evaluate \
  --trace trace.json --rubric rubric.json \
  --vlm-verdict vlm.json --output physics_result.json
```

The bundled PhyGenBench catalog preserves source polarity and reference conflicts. It is a reference-assisted adaptation, not the official multi-model PhyGenEval metric. Reference descriptions specify expected behavior and are never treated as observations.
