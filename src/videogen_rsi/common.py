import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        temp = f.name
    os.replace(temp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]


def validate_execution_environment(config, environ=None):
    """Check declared runtime settings without recording unrelated environment."""
    environment = os.environ if environ is None else environ
    execution = config.get("execution", {})
    key = "PYTORCH_CUDA_ALLOC_CONF"
    if "cuda_allocator_conf" in execution and environment.get(key, "") != execution["cuda_allocator_conf"]:
        raise ValueError("CUDA allocator environment differs from the frozen execution profile")


def parse_json(text):
    """Find a JSON object without accepting Python literals or executing model text."""
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[i:])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass
    raise ValueError("Model did not return a JSON object")


def parse_judgment(text, expected_ids):
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        checks = value.get("checks") if isinstance(value, dict) else value
        if not isinstance(checks, list) or not all(isinstance(c, dict) for c in checks):
            continue
        ids = [c.get("id") for c in checks]
        if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
            raise ValueError("Judge returned missing, unknown, or duplicate checklist IDs")
        return {"checks": checks}
    raise ValueError("Judge did not return a complete checklist object or array")


def validate_task(task):
    assert task["split"] in {"development", "calibration", "admission", "test"}
    assert task["id"] and task["prompt"] and task["family"]
    ids = [c["id"] for c in task["constraints"]]
    assert ids and len(ids) == len(set(ids))
    assert all(c["text"] for c in task["constraints"])
    dependencies = {c["id"]: c.get("requires", []) for c in task["constraints"]}
    def visit(node, ancestors):
        if node in ancestors:
            raise ValueError("Cyclic event requirements")
        for dep in dependencies[node]:
            if dep not in dependencies:
                raise ValueError("Unknown event prerequisite")
            visit(dep, ancestors | {node})
    for node in dependencies:
        visit(node, set())


def normalize_judgment(raw, task, duration):
    """Fail closed: missing/duplicate/malformed evidence never becomes a pass."""
    entries = raw.get("checks", [])
    if not isinstance(entries, list):
        entries = []
    checks = []
    for constraint in task["constraints"]:
        matches = [c for c in entries if isinstance(c, dict) and c.get("id") == constraint["id"]]
        item = dict(matches[0]) if len(matches) == 1 else {}
        status = item.get("status", "uncertain")
        evidence = item.get("evidence", "")
        interval = item.get("interval_s")
        valid_interval = (isinstance(interval, list) and len(interval) == 2 and
                          all(isinstance(v, (int, float)) and math.isfinite(v) for v in interval) and
                          0 <= interval[0] <= interval[1] <= duration + 0.1)
        confidence = item.get("confidence")
        if confidence is not None and (not isinstance(confidence, (int, float)) or not math.isfinite(confidence)):
            confidence = 0
        if confidence is not None:
            confidence = max(0, min(1, confidence))
        if status not in {"pass", "fail", "uncertain"} or not isinstance(evidence, str) or not evidence.strip():
            status = "uncertain"
        if not valid_interval or (confidence is not None and confidence < 0.6):
            status = "uncertain"
        checks.append({"id": constraint["id"], "text": constraint["text"], "status": status,
                       "evidence": str(evidence), "interval_s": interval if valid_interval else None,
                       "confidence": confidence,
                       "confidence_source": "model_reported_uncalibrated" if confidence is not None else "not_provided"})
    # A temporal clause cannot pass vacuously when its antecedent event failed.
    # Dependencies are prespecified task data, never inferred from test verdicts.
    by_id = {c["id"]:c for c in checks}
    requirements = {c["id"]:c.get("requires",[]) for c in task["constraints"]}
    resolved = set()
    def resolve(cid, ancestors):
        if cid in ancestors:
            raise ValueError("Cyclic event requirements")
        if cid in resolved:
            return
        check = by_id[cid]
        deps = requirements[cid]
        for dep in deps:
            if dep not in by_id:
                raise ValueError("Unknown event prerequisite")
            resolve(dep, ancestors | {cid})
        unmet = [dep for dep in deps if by_id[dep]["status"] != "pass"]
        if unmet and check["status"] == "pass":
            check["raw_status"] = "pass"
            check["status"] = "fail" if any(by_id[dep]["status"] == "fail" for dep in unmet) else "uncertain"
            check["dependency_failure_ids"] = unmet
        resolved.add(cid)
    for cid in by_id:
        resolve(cid,set())
    return {"checks": checks, "strict_pass": all(c["status"] == "pass" for c in checks),
            "score": sum(c["status"] == "pass" for c in checks) / len(checks),
            "uncertain_fraction": sum(c["status"] == "uncertain" for c in checks) / len(checks)}


def regressed_checks(before, after):
    previous = {c["id"] for c in before["checks"] if c["status"] == "pass"}
    current = {c["id"] for c in after["checks"] if c["status"] == "pass"}
    return sorted(previous - current)


def choose_best(records):
    # Stable tie-breaker favors the earlier sample; never look at external evaluation.
    return max(records, key=lambda r: (r["judgment"]["strict_pass"], r["judgment"]["score"],
                                       -r["judgment"]["uncertain_fraction"], -r["round"]))
