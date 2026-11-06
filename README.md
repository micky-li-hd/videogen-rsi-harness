# VideoGen-RSI Harness

An independently implemented outer loop for evolving controller artifacts around frozen video generators and evaluators. The design adapts the frozen-model harness-search paradigm of [Tencent/VideoHarness-RSI](https://github.com/Tencent/VideoHarness-RSI) to stochastic video generation.

The harness provides:

- sealed candidate artifacts and frozen evaluator/proposer commands;
- development-only candidate search with strict promotion and quality guards;
- blinded paired calibration and admission with task-cluster bootstrap gates;
- an exact-candidate lock before held-out test access;
- append-only hash-chained lifecycle records and fail-closed mutation checks;
- a bounded Codex proposer adapter and a zero-cost offline lifecycle fixture.
- fixed full-frame physical boundary rules for gravity, collisions, contact, friction, material behavior and event timing;
- reference-assisted per-prompt rubrics for all 160 PhyGenBench items;
- VLM/numerical arbitration with hard-rule vetoes and explicit abstention on missing measurements.

The included mock evaluator produces synthetic scores solely to validate orchestration. It is not benchmark evidence. Formal results require a frozen adapter for Wan generation and an independent video evaluator.

## Quick start

```bash
PYTHONPATH=src python -m videogen_rsi.harness_cli init \
  --config configs/videogen_rsi_harness_smoke_v1.json \
  --run runs/harness_smoke_v1
PYTHONPATH=src python -m videogen_rsi.harness_cli evolve \
  --run runs/harness_smoke_v1 --iterations 2
PYTHONPATH=src python -m videogen_rsi.harness_cli finalize \
  --run runs/harness_smoke_v1
PYTHONPATH=src python -m videogen_rsi.harness_cli test \
  --run runs/harness_smoke_v1
PYTHONPATH=src python -m videogen_rsi.harness_cli doctor \
  --run runs/harness_smoke_v1
```

Run the tests with:

```bash
PYTHONPATH=src python -m unittest tests.test_evolution_harness -v
```

See [the architecture guide](docs/VIDEOGEN_RSI_HARNESS.md) and [the upstream similarity/novelty audit](docs/VIDEOHARNESS_RSI_SIMILARITY_AUDIT.md).

The physics evaluator consumes an explicit trajectory trace; it does not infer exact geometry from pixels. Run it with `python -m videogen_rsi.physics_cli evaluate`. See `data/physics_evaluator_v1/README.md` for the schema and evidence boundary.

## Attribution

The high-level research pattern—keep the foundation model frozen, let a coding agent propose harness changes, promote on development results, and hide held-out test data—is explicitly attributed to VideoHarness-RSI. No upstream source file is copied into this repository. This implementation changes the candidate contract and validation protocol for paired, stochastic video generation.
