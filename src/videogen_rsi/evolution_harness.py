"""Durable outer loop for evolving a video-generation controller.

The design adapts the strict-frontier/search-ledger pattern of Tencent's
VideoHarness-RSI to generation.  The generator and evaluator commands are
frozen in the run config; a proposer may only create candidate artifacts.
Development drives search.  Calibration/admission gate finalization, and the
test split is unreachable until finalization succeeds.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from .common import digest, read_json, write_json
from .memory import admission_phase


SCHEMA_VERSION = 1
SEARCH_SPLIT = "development"
ADMISSION_SPLITS = ("calibration", "admission")
FINAL_SPLIT = "test"
DERIVED_CANDIDATE_KEYS = {"id", "artifact_sha256"}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {k: candidate[k] for k in sorted(candidate) if k not in DERIVED_CANDIDATE_KEYS}


def candidate_id(candidate: dict[str, Any]) -> str:
    return digest(canonical_candidate(candidate))


def validate_candidate(candidate: dict[str, Any], expected_parent: str | None) -> None:
    required = {"schema_version", "name", "parent_id", "hypothesis", "strategy"}
    if set(candidate) - (required | DERIVED_CANDIDATE_KEYS):
        raise ValueError("Candidate contains unknown top-level fields")
    if not required <= set(candidate) or candidate["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Candidate schema is incomplete or unsupported")
    if candidate["parent_id"] != expected_parent:
        raise ValueError("Candidate parent differs from the frozen frontier")
    if not all(isinstance(candidate[k], str) and candidate[k].strip() for k in ("name", "hypothesis")):
        raise ValueError("Candidate requires a nonempty name and hypothesis")
    if not isinstance(candidate["strategy"], dict) or not candidate["strategy"]:
        raise ValueError("Candidate strategy must be a nonempty object")


def seal_candidate(candidate: dict[str, Any], directory: Path, expected_parent: str | None) -> dict[str, Any]:
    validate_candidate(candidate, expected_parent)
    implementation = directory / "implementation"
    files = {}
    if implementation.exists():
        for path in sorted(implementation.rglob("*")):
            if path.is_symlink():
                raise ValueError("Candidate implementation may not contain symlinks")
            if path.is_file():
                files[str(path.relative_to(directory))] = file_sha256(path)
    sealed = {**canonical_candidate(candidate), "implementation_files": files}
    sealed["id"] = candidate_id(sealed)
    sealed["artifact_sha256"] = hashlib.sha256(
        json.dumps(sealed, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return sealed


def validate_config(config: dict[str, Any]) -> None:
    required = {"schema_version", "seed_candidate", "proposer", "evaluator", "promotion", "splits"}
    if set(config) != required or config["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Harness config keys or schema changed")
    for component in ("proposer", "evaluator"):
        value = config[component]
        if not isinstance(value, dict) or not isinstance(value.get("command"), list) or not value["command"]:
            raise ValueError(f"{component} requires a nonempty argv command")
        if not all(isinstance(x, str) and x for x in value["command"]):
            raise ValueError(f"{component} command must be a list of strings")
    if set(config["splits"]) != {SEARCH_SPLIT, *ADMISSION_SPLITS, FINAL_SPLIT}:
        raise ValueError("Exactly development/calibration/admission/test splits are required")
    promotion = config["promotion"]
    if not isinstance(promotion.get("primary_metric"), str) or promotion.get("min_delta", 0) < 0:
        raise ValueError("Promotion requires a primary metric and nonnegative min_delta")
    if not isinstance(promotion.get("quality_guards", []), list):
        raise ValueError("quality_guards must be a list")
    if promotion.get("require_paired_admission") is not True:
        raise ValueError("Production harness requires paired calibration/admission")


def expand_command(argv: list[str], values: dict[str, Any]) -> list[str]:
    rendered = []
    for token in argv:
        try:
            rendered.append(token.format_map({k: str(v) for k, v in values.items()}))
        except KeyError as exc:
            raise ValueError(f"Unknown command placeholder: {exc.args[0]}") from exc
    return rendered


def run_component(argv: list[str], *, cwd: Path, timeout: int, stdin: str | None = None) -> dict[str, Any]:
    started = time.time()
    try:
        result = subprocess.run(argv, cwd=cwd, input=stdin, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return {"status": "timeout", "returncode": 124, "elapsed_seconds": time.time() - started,
                "stdout": (exc.stdout or "")[-4000:], "stderr": (exc.stderr or "")[-4000:]}
    return {"status": "completed" if result.returncode == 0 else "failed", "returncode": result.returncode,
            "elapsed_seconds": time.time() - started, "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]}


class EvolutionHarness:
    def __init__(self, run_dir: str | Path):
        self.root = Path(run_dir).resolve()
        self.config_path = self.root / "frozen_config.json"
        if not self.config_path.is_file():
            raise ValueError(f"Harness is not initialized: {self.root}")
        self.config = read_json(self.config_path)
        validate_config(self.config)
        self.state_path = self.root / "state.json"
        self.state = read_json(self.state_path)

    @classmethod
    def initialize(cls, config_path: str | Path, run_dir: str | Path) -> "EvolutionHarness":
        config_path = Path(config_path).resolve()
        run_dir = Path(run_dir).resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError("Refusing to initialize into a nonempty run directory")
        config = read_json(config_path)
        validate_config(config)
        config["seed_candidate"] = str((config_path.parent / config["seed_candidate"]).resolve())
        for split in config["splits"].values():
            split["manifest"] = str((config_path.parent / split["manifest"]).resolve())
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "frozen_config.json", config)
        seed_source = Path(config["seed_candidate"])
        seed_raw = read_json(seed_source)
        seed_dir = run_dir / "candidates" / "seed"
        seed_dir.mkdir(parents=True)
        sealed = seal_candidate(seed_raw, seed_dir, None)
        final_dir = run_dir / "candidates" / sealed["id"]
        seed_dir.rename(final_dir)
        write_json(final_dir / "candidate.json", sealed)
        state = {"schema_version": SCHEMA_VERSION, "status": "search_ready", "iteration": 0,
                 "frontier_id": sealed["id"], "finalized_id": None, "test_result": None,
                 "config_sha256": file_sha256(run_dir / "frozen_config.json"), "created_at": time.time()}
        write_json(run_dir / "state.json", state)
        harness = cls(run_dir)
        harness._event("initialized", {"frontier_id": sealed["id"], "source_config": str(config_path)})
        return harness

    def _save_state(self) -> None:
        write_json(self.state_path, self.state)

    def _event(self, event: str, payload: dict[str, Any]) -> None:
        ledger = self.root / "ledger.jsonl"
        previous = None
        if ledger.exists():
            lines = [line for line in ledger.read_text().splitlines() if line.strip()]
            if lines:
                previous = json.loads(lines[-1])["event_sha256"]
        record = {"schema_version": SCHEMA_VERSION, "at": time.time(), "event": event,
                  "payload": payload, "previous_event_sha256": previous}
        record["event_sha256"] = hashlib.sha256(
            json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def candidate_path(self, candidate_id_: str) -> Path:
        path = self.root / "candidates" / candidate_id_ / "candidate.json"
        candidate = read_json(path)
        if candidate["id"] != candidate_id_ or candidate["id"] != candidate_id(candidate):
            raise ValueError("Candidate identity changed")
        implementation = path.parent / "implementation"
        observed = {}
        if implementation.exists():
            for item in sorted(implementation.rglob("*")):
                if item.is_file():
                    observed[str(item.relative_to(path.parent))] = file_sha256(item)
        if observed != candidate["implementation_files"]:
            raise ValueError("Candidate implementation changed after sealing")
        return path

    def _component_values(self, **extra: Any) -> dict[str, Any]:
        return {"python": sys.executable, "workspace": Path.cwd(), "run_dir": self.root, **extra}

    def _evaluate(self, candidate_id_: str, split: str) -> dict[str, Any]:
        if split == FINAL_SPLIT and self.state.get("finalized_id") != candidate_id_:
            raise ValueError("Test split is locked until this exact candidate is finalized")
        output = self.root / "evaluations" / split / f"{candidate_id_}.json"
        if output.exists():
            result = read_json(output)
            self._validate_result(result, candidate_id_, split)
            return result
        output.parent.mkdir(parents=True, exist_ok=True)
        candidate = self.candidate_path(candidate_id_)
        command = expand_command(self.config["evaluator"]["command"], self._component_values(
            candidate=candidate, candidate_dir=candidate.parent, split=split, output=output,
            split_manifest=Path(self.config["splits"][split]["manifest"]).resolve()))
        attempt = run_component(command, cwd=Path.cwd(), timeout=int(self.config["evaluator"].get("timeout_seconds", 86400)))
        write_json(output.with_suffix(".attempt.json"), {"command": command, **attempt})
        if attempt["status"] != "completed" or not output.is_file():
            raise RuntimeError(f"Evaluator failed for {candidate_id_}/{split}: {attempt['stderr'][-1000:]}")
        result = read_json(output)
        self._validate_result(result, candidate_id_, split)
        self._event("evaluated", {"candidate_id": candidate_id_, "split": split,
                                  "result_sha256": file_sha256(output), "aggregate": result["aggregate"]})
        return result

    def _validate_result(self, result: dict[str, Any], candidate_id_: str, split: str) -> None:
        if set(result) != {"schema_version", "candidate_id", "split", "records", "aggregate", "cost"}:
            raise ValueError("Evaluator result schema changed")
        if result["schema_version"] != SCHEMA_VERSION or result["candidate_id"] != candidate_id_ or result["split"] != split:
            raise ValueError("Evaluator result binding changed")
        records = result["records"]
        if not isinstance(records, list) or not records:
            raise ValueError("Evaluator must retain every unit as a record")
        keys = [(r.get("task_id"), r.get("seed")) for r in records]
        if len(keys) != len(set(keys)) or any(not task or type(seed) is not int for task, seed in keys):
            raise ValueError("Evaluation units are missing or duplicated")
        expected = self.config["splits"][split].get("expected_units")
        if expected is not None and len(records) != expected:
            raise ValueError(f"Expected {expected} {split} units, got {len(records)}")
        aggregate = result["aggregate"]
        if not isinstance(aggregate, dict) or any(type(v) not in (int, float) for v in aggregate.values()):
            raise ValueError("Aggregate metrics must be finite numbers")
        if any(not (-1e100 < float(v) < 1e100) for v in aggregate.values()):
            raise ValueError("Aggregate metric is not finite")
        if split in ADMISSION_SPLITS and self.config["promotion"]["require_paired_admission"]:
            for row in records:
                if any(type(row.get(k)) not in (int, float) for k in ("with_score", "without_score")):
                    raise ValueError("Admission records require paired with_score/without_score")

    def _promotion(self, parent: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        policy = self.config["promotion"]
        primary = policy["primary_metric"]
        if primary not in parent["aggregate"] or primary not in candidate["aggregate"]:
            raise ValueError("Primary metric missing")
        delta = candidate["aggregate"][primary] - parent["aggregate"][primary]
        strict = delta > policy.get("min_delta", 0)
        guards = {}
        for guard in policy.get("quality_guards", []):
            metric = guard["metric"]
            floor = parent["aggregate"][metric] + guard.get("min_delta", 0)
            guards[metric] = candidate["aggregate"][metric] >= floor
        return strict and all(guards.values()), {"primary_metric": primary, "delta": delta,
                                                 "strict_primary_gain": strict, "quality_guards": guards}

    def evolve(self, iterations: int = 1) -> dict[str, Any]:
        if self.state["status"] != "search_ready":
            raise ValueError("Evolution is closed after finalization/test")
        if iterations < 1:
            raise ValueError("iterations must be positive")
        for _ in range(iterations):
            parent_id = self.state["frontier_id"]
            parent_result = self._evaluate(parent_id, SEARCH_SPLIT)
            iteration = self.state["iteration"] + 1
            iteration_dir = self.root / "iterations" / f"{iteration:04d}"
            candidate_dir = iteration_dir / "candidate"
            candidate_dir.mkdir(parents=True, exist_ok=False)
            proposal_output = iteration_dir / "proposal.json"
            request = {"schema_version": SCHEMA_VERSION, "iteration": iteration,
                       "frontier": read_json(self.candidate_path(parent_id)),
                       "frontier_development_result": parent_result,
                       "search_contract": {"development_only": True, "test_hidden": True,
                                           "strict_primary_gain": True,
                                           "quality_guards": self.config["promotion"].get("quality_guards", [])},
                       "candidate_directory": str(candidate_dir),
                       "required_output": str(proposal_output)}
            request_path = iteration_dir / "proposal_request.json"
            write_json(request_path, request)
            command = expand_command(self.config["proposer"]["command"], self._component_values(
                request=request_path, output=proposal_output, candidate_dir=candidate_dir, iteration=iteration,
                parent=self.candidate_path(parent_id)))
            attempt = run_component(command, cwd=Path.cwd(), timeout=int(self.config["proposer"].get("timeout_seconds", 3600)))
            write_json(iteration_dir / "proposal_attempt.json", {"command": command, **attempt})
            if attempt["status"] != "completed" or not proposal_output.is_file():
                self._event("proposal_failed", {"iteration": iteration, "parent_id": parent_id,
                                                "status": attempt["status"], "returncode": attempt["returncode"]})
                self.state["iteration"] = iteration
                self._save_state()
                continue
            sealed = seal_candidate(read_json(proposal_output), candidate_dir, parent_id)
            final_dir = self.root / "candidates" / sealed["id"]
            if final_dir.exists():
                raise ValueError("Proposer repeated an existing candidate")
            candidate_dir.rename(final_dir)
            write_json(final_dir / "candidate.json", sealed)
            candidate_result = self._evaluate(sealed["id"], SEARCH_SPLIT)
            accepted, decision = self._promotion(parent_result, candidate_result)
            decision.update({"iteration": iteration, "parent_id": parent_id, "candidate_id": sealed["id"],
                             "accepted": accepted})
            write_json(iteration_dir / "decision.json", decision)
            self._event("candidate_decided", decision)
            if accepted:
                self.state["frontier_id"] = sealed["id"]
            self.state["iteration"] = iteration
            self._save_state()
        return self.status()

    def finalize(self) -> dict[str, Any]:
        if self.state["status"] != "search_ready":
            raise ValueError("Harness is not open for finalization")
        candidate_id_ = self.state["frontier_id"]
        phases = {}
        for split in ADMISSION_SPLITS:
            result = self._evaluate(candidate_id_, split)
            trials = [{"task_id": row["task_id"], "seed": row["seed"], "split": split,
                       "with_score": row["with_score"], "without_score": row["without_score"],
                       "blinded": row.get("blinded") is True, "evaluator_role": row.get("evaluator_role"),
                       "evaluator_id": row.get("evaluator_id")} for row in result["records"]]
            phases[split] = admission_phase(
                trials, split,
                min_tasks=int(self.config["splits"][split].get("min_tasks", 4)),
                min_seeds=int(self.config["splits"][split].get("min_seeds", 2)),
                bootstrap_samples=int(self.config["splits"][split].get("bootstrap_samples", 5000)))
        passed = all(v["passed"] for v in phases.values())
        record = {"schema_version": SCHEMA_VERSION, "candidate_id": candidate_id_, "phases": phases,
                  "passed": passed, "at": time.time()}
        write_json(self.root / "finalization.json", record)
        self._event("finalization_decided", record)
        if passed:
            self.state.update(status="finalized", finalized_id=candidate_id_)
            self._save_state()
        return record

    def test(self) -> dict[str, Any]:
        candidate_id_ = self.state.get("finalized_id")
        if self.state["status"] != "finalized" or not candidate_id_:
            raise ValueError("Test remains locked until calibration and admission both pass")
        if self.state.get("test_result"):
            return read_json(self.state["test_result"])
        result = self._evaluate(candidate_id_, FINAL_SPLIT)
        relative = str((self.root / "evaluations" / FINAL_SPLIT / f"{candidate_id_}.json").relative_to(self.root))
        self.state.update(status="complete", test_result=relative)
        self._save_state()
        self._event("test_completed", {"candidate_id": candidate_id_, "result": relative,
                                       "aggregate": result["aggregate"]})
        return result

    def doctor(self) -> dict[str, Any]:
        checks = {"config_unchanged": file_sha256(self.config_path) == self.state["config_sha256"],
                  "frontier_valid": False, "proposer_executable": False, "evaluator_executable": False,
                  "test_lock": self.state["status"] in {"finalized", "complete"} or self.state.get("test_result") is None}
        try:
            self.candidate_path(self.state["frontier_id"])
            checks["frontier_valid"] = True
        except Exception:
            pass
        values = self._component_values(request="x", output="x", candidate="x", candidate_dir="x",
                                        split="development", split_manifest="x", iteration=0, parent="x")
        for name in ("proposer", "evaluator"):
            command = expand_command(self.config[name]["command"], values)
            executable = command[0]
            checks[name + "_executable"] = Path(executable).is_file() or shutil.which(executable) is not None
        return {"passed": all(checks.values()), "checks": checks, "state": self.status()}

    def status(self) -> dict[str, Any]:
        return {k: self.state.get(k) for k in ("status", "iteration", "frontier_id", "finalized_id", "test_result")}

