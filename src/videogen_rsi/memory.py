"""Experience admission requires independent, blinded paired observations.

Development wins propose candidates. They never automatically activate memory.
Bootstrap resampling is clustered by task, not falsely independent seed rows.
"""
from collections import defaultdict
import random
from .common import digest


STRATEGY_FIELDS = ("name", "trigger_families", "negative_triggers", "action", "side_effects")


def candidate_id(candidate):
    return digest({key: candidate[key] for key in STRATEGY_FIELDS})


def admission_phase(trials, split, min_tasks=4, min_seeds=2, bootstrap_samples=5000):
    eligible = [r for r in trials if r["split"] == split]
    by_task = defaultdict(list)
    seen = set()
    for row in eligible:
        key = (row["task_id"], row["seed"])
        if key in seen:
            raise ValueError("Duplicate paired trial")
        seen.add(key)
        if not row.get("blinded") or row.get("evaluator_role") != "independent":
            raise ValueError("Admission requires independent blinded evaluation")
        if not row.get("evaluator_id"):
            raise ValueError("Missing evaluator identity/version")
        for field in ("with_score", "without_score"):
            value = row[field]
            if not isinstance(value, (int,float)) or not 0 <= value <= 1:
                raise ValueError("Scores must be finite numbers in [0,1]")
        by_task[row["task_id"]].append(row["with_score"]-row["without_score"])
    enough = len(by_task) >= min_tasks and all(len(v) >= min_seeds for v in by_task.values())
    means = [sum(v)/len(v) for v in by_task.values()]
    if not enough:
        return {"passed": False, "reason": "insufficient independent tasks/seeds", "tasks": len(by_task)}
    rng = random.Random(20261027)
    boot = sorted(sum(rng.choice(means) for _ in means)/len(means) for _ in range(bootstrap_samples))
    lower, upper = boot[int(.025*bootstrap_samples)], boot[int(.975*bootstrap_samples)]
    regression = sum(row["with_score"] < row["without_score"] for row in eligible)/len(eligible)
    mean_gain = sum(means)/len(means)
    return {"passed": lower > 0 and regression <= .25,
            "tasks": len(means), "trials": len(eligible), "task_mean_gain": mean_gain,
            "task_cluster_bootstrap_95ci": [lower,upper], "regression_fraction": regression,
            "criteria": "CI lower > 0, regression fraction <= 0.25; small-sample exploratory gate"}


def admit(candidate, trials):
    if candidate["id"] != candidate_id(candidate):
        raise ValueError("Candidate content changed without a new version")
    development = set(candidate["source_task_ids"])
    calibration = {r["task_id"] for r in trials if r["split"] == "calibration"}
    admission = {r["task_id"] for r in trials if r["split"] == "admission"}
    if development & (calibration|admission) or calibration & admission:
        raise ValueError("Experience extraction/calibration/admission task leakage")
    if any(r["split"] not in {"calibration","admission"} for r in trials):
        raise ValueError("Test or development verdicts cannot admit memory")
    if any(r.get("candidate_id") != candidate["id"] for r in trials):
        raise ValueError("Candidate-version mismatch")
    phases = {split: admission_phase(trials,split) for split in ("calibration","admission")}
    status = "active" if all(p["passed"] for p in phases.values()) else "candidate"
    return {**candidate, "status": status, "admission": phases, "trials_hash": digest(trials)}


def retrieve(library, task):
    return [e for e in library if e.get("status") == "active" and task["family"] in e["trigger_families"]]
