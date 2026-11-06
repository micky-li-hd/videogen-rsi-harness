"""Offline deterministic component used only to verify the harness lifecycle."""

import argparse
import json
from pathlib import Path

from videogen_rsi.common import digest, write_json


def propose(a):
    request = json.loads(Path(a.request).read_text()); parent = request["frontier"]
    score = float(parent["strategy"].get("fixture_score", 0)) + 0.1
    write_json(a.output, {"schema_version": 1, "name": f"fixture-{request['iteration']}", "parent_id": parent["id"],
                         "hypothesis": "Offline lifecycle fixture increments a declared score.",
                         "strategy": {"fixture_score": score, "diagnostic": "fixed", "repair": "fixed",
                                      "generation_budget": 2, "selection": "fixed"}})


def evaluate(a):
    candidate = json.loads(Path(a.candidate).read_text()); score = float(candidate["strategy"].get("fixture_score", 0))
    records = []
    for task in range(4):
        for seed in (1, 2):
            row = {"task_id": f"{a.split}_{task}", "seed": seed, "status": "completed"}
            if a.split in {"calibration", "admission"}:
                row.update(with_score=min(1.0, .6 + score), without_score=.5, blinded=True,
                           evaluator_role="independent", evaluator_id="offline-fixture-evaluator")
            records.append(row)
    write_json(a.output, {"schema_version": 1, "candidate_id": candidate["id"], "split": a.split,
                          "records": records, "aggregate": {"primary": score, "identity": 1.0, "naturalness": 1.0},
                          "cost": {"generation_calls": 0, "API_calls": 0, "fixture": True}})


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("propose"); q.add_argument("--request",required=True); q.add_argument("--output",required=True)
    q.add_argument("--candidate-dir",required=True)
    e=sub.add_parser("evaluate"); e.add_argument("--candidate",required=True); e.add_argument("--split",required=True)
    e.add_argument("--manifest",required=True); e.add_argument("--output",required=True)
    a=p.parse_args(); propose(a) if a.command=="propose" else evaluate(a)


if __name__ == "__main__": main()
