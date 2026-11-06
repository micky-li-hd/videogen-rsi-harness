from pathlib import Path
import json
import tempfile
import unittest

from videogen_rsi.common import read_json
from videogen_rsi.evolution_harness import EvolutionHarness


ROOT = Path(__file__).resolve().parents[1]


class EvolutionHarnessTests(unittest.TestCase):
    def test_full_lifecycle_strict_frontier_and_test_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            harness = EvolutionHarness.initialize(ROOT / "configs/videogen_rsi_harness_smoke_v1.json", run)
            with self.assertRaises(ValueError): harness.test()
            first = harness.state["frontier_id"]
            harness.evolve(2)
            self.assertEqual(harness.state["iteration"], 2)
            self.assertNotEqual(harness.state["frontier_id"], first)
            finalization = harness.finalize()
            self.assertTrue(finalization["passed"])
            result = harness.test()
            self.assertEqual(result["split"], "test")
            self.assertEqual(harness.status()["status"], "complete")
            self.assertTrue(harness.doctor()["passed"])
            events = [json.loads(line) for line in (run / "ledger.jsonl").read_text().splitlines()]
            self.assertTrue(events)
            for previous, current in zip(events, events[1:]):
                self.assertEqual(current["previous_event_sha256"], previous["event_sha256"])

    def test_candidate_and_config_mutation_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            harness = EvolutionHarness.initialize(ROOT / "configs/videogen_rsi_harness_smoke_v1.json", run)
            candidate = harness.candidate_path(harness.state["frontier_id"])
            value = read_json(candidate); value["strategy"]["fixture_score"] = 99
            candidate.write_text(json.dumps(value))
            with self.assertRaises(ValueError): harness.candidate_path(harness.state["frontier_id"])


if __name__ == "__main__": unittest.main()
