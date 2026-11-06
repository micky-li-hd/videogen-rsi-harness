"""Deterministic physical-boundary evaluation and VLM/numeric arbitration.

The evaluator consumes externally extracted full-frame tracks. It never infers
coordinates from pixels and never treats a reference rubric as an observation.
Hard-rule failures veto VLM opinions; missing measurements remain ``unknown``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Any


STATUSES = {"pass", "fail", "unknown"}
RULE_TYPES = {
    "full_frame_coverage", "gravity_direction", "acceleration_range",
    "velocity_continuity", "collision_momentum", "restitution",
    "contact", "penetration", "hover", "friction_stop", "sliding_stop",
    "material_boundary", "event_order", "event_time_window",
}


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _sub(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _scale(value: float, vector: list[float]) -> list[float]:
    return [value * x for x in vector]


def validate_trace(trace: dict[str, Any]) -> dict[str, Any]:
    required = {"schema_version", "task_id", "fps", "frame_count", "objects", "events"}
    if set(trace) != required or trace["schema_version"] != 1:
        raise ValueError("Trace requires the exact v1 schema")
    if not isinstance(trace["task_id"], str) or not trace["task_id"]:
        raise ValueError("Trace task_id is required")
    if not _finite(trace["fps"]) or trace["fps"] <= 0:
        raise ValueError("Trace fps must be positive")
    if type(trace["frame_count"]) is not int or trace["frame_count"] < 2:
        raise ValueError("Trace frame_count must be at least two")
    if not isinstance(trace["objects"], list) or not trace["objects"]:
        raise ValueError("Trace requires tracked objects")
    object_ids = [row.get("id") for row in trace["objects"]]
    if any(not isinstance(value, str) or not value for value in object_ids) or len(object_ids) != len(set(object_ids)):
        raise ValueError("Object IDs must be unique nonempty strings")
    for obj in trace["objects"]:
        if set(obj) != {"id", "material", "mass", "samples"}:
            raise ValueError("Object requires id/material/mass/samples")
        if obj["material"] not in {"rigid", "soft", "liquid", "unknown"}:
            raise ValueError("Unsupported material category")
        if obj["mass"] is not None and (not _finite(obj["mass"]) or obj["mass"] <= 0):
            raise ValueError("Mass must be positive or null")
        if not isinstance(obj["samples"], list):
            raise ValueError("Samples must be a list")
        frames = []
        for sample in obj["samples"]:
            if set(sample) != {"frame", "center", "visible", "signed_distances", "metrics"}:
                raise ValueError("Sample requires frame/center/visible/signed_distances/metrics")
            frame = sample["frame"]
            if type(frame) is not int or not 0 <= frame < trace["frame_count"]:
                raise ValueError("Sample frame is outside the decoded video")
            if type(sample["visible"]) is not bool:
                raise ValueError("Sample visibility must be boolean")
            center = sample["center"]
            if center is not None and (not isinstance(center, list) or len(center) not in (2, 3) or not all(_finite(x) for x in center)):
                raise ValueError("Center must be a finite 2D/3D vector or null")
            if not isinstance(sample["signed_distances"], dict) or not isinstance(sample["metrics"], dict):
                raise ValueError("Distances and metrics must be objects")
            if any(not isinstance(k, str) or not _finite(v) for k, v in sample["signed_distances"].items()):
                raise ValueError("Signed distances must be finite")
            if any(not isinstance(k, str) or not _finite(v) for k, v in sample["metrics"].items()):
                raise ValueError("Metrics must be finite")
            frames.append(frame)
        if frames != sorted(set(frames)):
            raise ValueError("Object samples must have distinct increasing frames")
    if not isinstance(trace["events"], list):
        raise ValueError("Events must be a list")
    event_ids = []
    for event in trace["events"]:
        if set(event) != {"id", "type", "frame", "objects"}:
            raise ValueError("Event requires id/type/frame/objects")
        if not isinstance(event["id"], str) or not event["id"] or not isinstance(event["type"], str):
            raise ValueError("Event identity and type are required")
        if type(event["frame"]) is not int or not 0 <= event["frame"] < trace["frame_count"]:
            raise ValueError("Event frame is outside the decoded video")
        if not isinstance(event["objects"], list) or any(x not in object_ids for x in event["objects"]):
            raise ValueError("Event references unknown objects")
        event_ids.append(event["id"])
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Event IDs must be unique")
    return trace


def validate_rubric(rubric: dict[str, Any]) -> dict[str, Any]:
    required = {"schema_version", "rubric_id", "task_id", "source", "reference_flags", "checks", "hard_rules"}
    if set(rubric) != required or rubric["schema_version"] != 1:
        raise ValueError("Rubric requires the exact v1 schema")
    if not all(isinstance(rubric[k], str) and rubric[k] for k in ("rubric_id", "task_id", "source")):
        raise ValueError("Rubric identity is required")
    if not isinstance(rubric["reference_flags"], list) or any(
        not isinstance(row, dict) or row.get("severity") not in {"warning", "caveat", "conflict"}
        for row in rubric["reference_flags"]
    ):
        raise ValueError("Reference flags must preserve warning/conflict severity")
    rules = rubric["hard_rules"]
    if not isinstance(rules, list):
        raise ValueError("hard_rules must be a list")
    rule_ids = []
    for rule in rules:
        if set(rule) != {"id", "type", "params"} or rule["type"] not in RULE_TYPES or not isinstance(rule["params"], dict):
            raise ValueError("Invalid hard rule")
        rule_ids.append(rule["id"])
    if any(not isinstance(x, str) or not x for x in rule_ids) or len(rule_ids) != len(set(rule_ids)):
        raise ValueError("Hard-rule IDs must be unique")
    if not isinstance(rubric["checks"], list) or not rubric["checks"]:
        raise ValueError("Rubric checks are required")
    check_ids = []
    for check in rubric["checks"]:
        if set(check) != {"id", "description", "required", "hard_rule_ids", "vlm_check_ids"}:
            raise ValueError("Invalid rubric check")
        if type(check["required"]) is not bool or not isinstance(check["description"], str) or not check["description"]:
            raise ValueError("Check description/required flag is invalid")
        if not isinstance(check["hard_rule_ids"], list) or not isinstance(check["vlm_check_ids"], list):
            raise ValueError("Check channel IDs must be lists")
        if any(x not in rule_ids for x in check["hard_rule_ids"]):
            raise ValueError("Check references an unknown hard rule")
        if len(check["hard_rule_ids"]) != len(set(check["hard_rule_ids"])) or len(check["vlm_check_ids"]) != len(set(check["vlm_check_ids"])):
            raise ValueError("Check channel IDs must be distinct")
        if not check["hard_rule_ids"] and not check["vlm_check_ids"]:
            raise ValueError("Every check needs numeric or VLM evidence")
        check_ids.append(check["id"])
    if len(check_ids) != len(set(check_ids)):
        raise ValueError("Rubric check IDs must be unique")
    return rubric


def _object(trace: dict[str, Any], object_id: str) -> dict[str, Any]:
    rows = [row for row in trace["objects"] if row["id"] == object_id]
    if len(rows) != 1:
        raise ValueError(f"Unknown object: {object_id}")
    return rows[0]


def _window_samples(trace: dict[str, Any], object_id: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    samples = [row for row in _object(trace, object_id)["samples"] if row["visible"] and row["center"] is not None]
    start, end = params.get("frame_window", [0, trace["frame_count"] - 1])
    if type(start) is not int or type(end) is not int or not 0 <= start <= end < trace["frame_count"]:
        raise ValueError("Invalid frame_window")
    return [row for row in samples if start <= row["frame"] <= end]


def _velocities(samples: list[dict[str, Any]], fps: float) -> list[tuple[int, list[float]]]:
    rows = []
    for left, right in zip(samples, samples[1:]):
        dt = (right["frame"] - left["frame"]) / fps
        if dt > 0:
            rows.append((right["frame"], [(b - a) / dt for a, b in zip(left["center"], right["center"])]))
    return rows


def _accelerations(velocities: list[tuple[int, list[float]]], fps: float) -> list[tuple[int, list[float]]]:
    rows = []
    for (lf, left), (rf, right) in zip(velocities, velocities[1:]):
        dt = (rf - lf) / fps
        if dt > 0:
            rows.append((rf, [(b - a) / dt for a, b in zip(left, right)]))
    return rows


def _result(rule: dict[str, Any], status: str, evidence: str, frames: list[int], measurements: dict[str, Any]) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError("Invalid rule status")
    return {"rule_id": rule["id"], "rule_type": rule["type"], "status": status,
            "evidence": evidence, "frame_indices": sorted(set(frames)), "measurements": measurements}


def _event(trace: dict[str, Any], event_id: str) -> dict[str, Any] | None:
    return next((row for row in trace["events"] if row["id"] == event_id), None)


def _velocity_near(trace: dict[str, Any], object_id: str, event_frame: int, side: str, window: int) -> list[float] | None:
    samples = _window_samples(trace, object_id, {"frame_window": [max(0, event_frame - window), min(trace["frame_count"] - 1, event_frame + window)]})
    velocities = _velocities(samples, trace["fps"])
    selected = [v for frame, v in velocities if (frame <= event_frame if side == "before" else frame > event_frame)]
    if not selected:
        return None
    return [median(row[axis] for row in selected) for axis in range(len(selected[0]))]


def evaluate_hard_rule(trace: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    validate_trace(trace)
    if rule.get("type") not in RULE_TYPES or set(rule) != {"id", "type", "params"}:
        raise ValueError("Invalid hard rule")
    p, kind = rule["params"], rule["type"]
    if kind == "full_frame_coverage":
        obj = _object(trace, p["object_id"]); observed = {row["frame"] for row in obj["samples"] if row["visible"] and row["center"] is not None}
        fraction = len(observed) / trace["frame_count"]; threshold = float(p.get("min_fraction", 1.0))
        return _result(rule, "pass" if fraction >= threshold else "fail", f"Observed {len(observed)}/{trace['frame_count']} decoded frames.", sorted(observed), {"coverage_fraction": fraction, "minimum": threshold})
    if kind in {"gravity_direction", "acceleration_range", "velocity_continuity"}:
        samples = _window_samples(trace, p["object_id"], p); velocities = _velocities(samples, trace["fps"]); accelerations = _accelerations(velocities, trace["fps"])
        frames = [row["frame"] for row in samples]
        if kind == "velocity_continuity":
            if len(velocities) < 2:
                return _result(rule, "unknown", "Fewer than two velocity estimates.", frames, {})
            excluded = {event["frame"] for event in trace["events"] if event["type"] in p.get("exclude_event_types", [])}
            jumps = [(frame, _norm(_sub(right, left))) for (frame, right), (_, left) in zip(velocities[1:], velocities[:-1]) if all(abs(frame-x) > p.get("event_margin_frames", 1) for x in excluded)]
            if not jumps:
                return _result(rule, "unknown", "All velocity transitions overlap excluded events.", frames, {})
            maximum = max(value for _, value in jumps); limit = float(p["max_delta_speed"])
            return _result(rule, "pass" if maximum <= limit else "fail", f"Maximum non-event speed jump is {maximum:.6g}.", [f for f, _ in jumps], {"max_delta_speed": maximum, "limit": limit})
        if not accelerations:
            return _result(rule, "unknown", "Fewer than three tracked positions.", frames, {})
        axis = int(p.get("axis", 1)); values = [vector[axis] for _, vector in accelerations]; estimate = median(values)
        if kind == "gravity_direction":
            sign = int(p.get("expected_sign", 1)); minimum = float(p.get("min_magnitude", 0.0)); passed = estimate * sign >= minimum
            return _result(rule, "pass" if passed else "fail", f"Median acceleration on axis {axis} is {estimate:.6g}.", [f for f, _ in accelerations], {"median_acceleration": estimate, "expected_sign": sign, "min_magnitude": minimum})
        lower, upper = map(float, p["range"]); passed = lower <= estimate <= upper
        return _result(rule, "pass" if passed else "fail", f"Median acceleration is {estimate:.6g}; required [{lower}, {upper}].", [f for f, _ in accelerations], {"median_acceleration": estimate, "range": [lower, upper]})
    if kind in {"collision_momentum", "restitution"}:
        event = _event(trace, p["event_id"])
        if event is None:
            return _result(rule, "unknown", "Required collision event is absent.", [], {})
        object_ids = p["object_ids"]
        if len(object_ids) != 2:
            raise ValueError("Collision rules require two objects")
        window = int(p.get("window_frames", 3)); before = [_velocity_near(trace, x, event["frame"], "before", window) for x in object_ids]; after = [_velocity_near(trace, x, event["frame"], "after", window) for x in object_ids]
        if any(value is None for value in before + after):
            return _result(rule, "unknown", "Collision lacks pre/post velocity evidence.", [event["frame"]], {})
        if kind == "collision_momentum":
            masses = [_object(trace, x)["mass"] for x in object_ids]
            if any(value is None for value in masses):
                return _result(rule, "unknown", "Object mass is missing.", [event["frame"]], {})
            pre = [sum(masses[i] * before[i][axis] for i in range(2)) for axis in range(len(before[0]))]
            post = [sum(masses[i] * after[i][axis] for i in range(2)) for axis in range(len(after[0]))]
            error = _norm(_sub(post, pre)) / max(_norm(pre), float(p.get("epsilon", 1e-6))); limit = float(p["relative_tolerance"])
            return _result(rule, "pass" if error <= limit else "fail", f"Relative momentum error is {error:.6g}.", [event["frame"]], {"momentum_before": pre, "momentum_after": post, "relative_error": error, "limit": limit})
        normal = [float(x) for x in p["normal"]]; length = _norm(normal)
        if length <= 0:
            raise ValueError("Collision normal must be nonzero")
        normal = [x / length for x in normal]
        pre_relative = _dot(_sub(before[1], before[0]), normal); post_relative = _dot(_sub(after[1], after[0]), normal)
        if pre_relative >= -float(p.get("epsilon", 1e-6)):
            return _result(rule, "unknown", "Objects are not observably approaching along the declared normal.", [event["frame"]], {"pre_relative_normal_speed": pre_relative})
        coefficient = -post_relative / pre_relative; lower, upper = map(float, p.get("range", [0.0, 1.0])); passed = lower <= coefficient <= upper
        return _result(rule, "pass" if passed else "fail", f"Estimated restitution coefficient is {coefficient:.6g}.", [event["frame"]], {"coefficient": coefficient, "range": [lower, upper]})
    if kind in {"contact", "penetration", "hover"}:
        samples = _object(trace, p["object_id"])["samples"]; target = p["target_id"]
        start, end = p.get("frame_window", [0, trace["frame_count"] - 1]); rows = [(s["frame"], s["signed_distances"].get(target)) for s in samples if start <= s["frame"] <= end and s["visible"] and target in s["signed_distances"]]
        if not rows:
            return _result(rule, "unknown", "No signed-distance samples for the target.", [], {})
        values = [value for _, value in rows]; tolerance = float(p.get("tolerance", 0.0))
        if kind == "contact": passed = min(abs(value) for value in values) <= tolerance; measurement = min(abs(value) for value in values); label = "minimum absolute gap"
        elif kind == "penetration": passed = min(values) >= -tolerance; measurement = min(values); label = "minimum signed distance"
        else: passed = max(values) <= tolerance; measurement = max(values); label = "maximum support gap"
        return _result(rule, "pass" if passed else "fail", f"{label.capitalize()} is {measurement:.6g}.", [f for f, _ in rows], {label.replace(' ', '_'): measurement, "tolerance": tolerance})
    if kind in {"friction_stop", "sliding_stop"}:
        samples = _window_samples(trace, p["object_id"], p)
        target = p.get("contact_target_id")
        if target is not None:
            tolerance = float(p.get("contact_tolerance", 0.0))
            available = [s for s in samples if target in s["signed_distances"]]
            if len(available) != len(samples):
                return _result(rule, "unknown", "Contact evidence is incomplete during the friction interval.", [s["frame"] for s in available], {})
            samples = [s for s in samples if abs(s["signed_distances"][target]) <= tolerance]
            if len(samples) < 3:
                return _result(rule, "fail", "The object is not continuously in contact during the declared friction interval.", [s["frame"] for s in available], {"contact_frames": len(samples), "observed_frames": len(available)})
        velocities = _velocities(samples, trace["fps"]); speeds = [(frame, _norm(value)) for frame, value in velocities]
        if len(speeds) < 2:
            return _result(rule, "unknown", "Insufficient velocities for sliding/friction.", [s["frame"] for s in samples], {})
        final_limit = float(p["final_speed_max"]); final_speed = speeds[-1][1]; initial_speed = speeds[0][1]
        if kind == "friction_stop":
            fraction = sum(right <= left + float(p.get("increase_tolerance", 0.0)) for (_, left), (_, right) in zip(speeds, speeds[1:])) / (len(speeds)-1); minimum = float(p.get("min_nonincreasing_fraction", 0.8)); passed = final_speed <= final_limit and fraction >= minimum
            return _result(rule, "pass" if passed else "fail", f"Speed changed {initial_speed:.6g}→{final_speed:.6g}; nonincreasing fraction {fraction:.3f}.", [f for f, _ in speeds], {"initial_speed": initial_speed, "final_speed": final_speed, "nonincreasing_fraction": fraction, "required_fraction": minimum})
        minimum_initial = float(p.get("initial_speed_min", 0.0)); passed = initial_speed >= minimum_initial and final_speed <= final_limit
        return _result(rule, "pass" if passed else "fail", f"Sliding speed changed {initial_speed:.6g}→{final_speed:.6g}.", [f for f, _ in speeds], {"initial_speed": initial_speed, "final_speed": final_speed, "initial_min": minimum_initial, "final_max": final_limit})
    if kind == "material_boundary":
        obj = _object(trace, p["object_id"]); material = p.get("material", obj["material"])
        if obj["material"] != material:
            return _result(rule, "fail", f"Declared material {material} disagrees with trace material {obj['material']}.", [], {})
        samples = [s for s in obj["samples"] if s["visible"]]; frames = [s["frame"] for s in samples]
        def series(name): return [s["metrics"][name] for s in samples if name in s["metrics"]]
        if material == "rigid":
            areas, aspects, deformation = series("area"), series("aspect_ratio"), series("deformation")
            if len(areas) != len(samples) or len(aspects) != len(samples): return _result(rule, "unknown", "Rigid-body area/aspect metrics are incomplete.", frames, {})
            def relspan(v): return (max(v)-min(v))/max(abs(median(v)), 1e-9)
            area_span, aspect_span = relspan(areas), relspan(aspects); deform = max(deformation) if deformation else 0.0
            passed = area_span <= float(p.get("max_area_rel_span", .15)) and aspect_span <= float(p.get("max_aspect_rel_span", .15)) and deform <= float(p.get("max_deformation", .15))
            return _result(rule, "pass" if passed else "fail", "Rigid-body shape invariants evaluated over the full track.", frames, {"area_rel_span": area_span, "aspect_rel_span": aspect_span, "max_deformation": deform})
        if material == "soft":
            deformation = series("deformation")
            if len(deformation) != len(samples) or not deformation: return _result(rule, "unknown", "Soft-body deformation metrics are incomplete.", frames, {})
            peak, final = max(deformation), deformation[-1]; passed = float(p.get("min_peak_deformation", 0)) <= peak <= float(p.get("max_peak_deformation", 1)) and final <= float(p.get("max_final_deformation", peak))
            return _result(rule, "pass" if passed else "fail", "Soft-body deformation and recovery evaluated.", frames, {"peak_deformation": peak, "final_deformation": final})
        if material == "liquid":
            volume, components = series("volume_proxy"), series("components")
            if len(volume) != len(samples) or not volume: return _result(rule, "unknown", "Liquid volume proxy is incomplete.", frames, {})
            conservation = (max(volume)-min(volume))/max(abs(median(volume)), 1e-9); maximum_components = max(components) if components else 1
            passed = conservation <= float(p.get("max_volume_rel_span", .25)) and maximum_components <= int(p.get("max_components", 20))
            return _result(rule, "pass" if passed else "fail", "Liquid conservation and fragmentation evaluated.", frames, {"volume_rel_span": conservation, "max_components": maximum_components})
        return _result(rule, "unknown", "Material is unknown.", frames, {})
    if kind in {"event_order", "event_time_window"}:
        if kind == "event_order":
            ids = p["event_ids"]; events = [_event(trace, value) for value in ids]
            if any(value is None for value in events): return _result(rule, "unknown", "One or more ordered events are absent.", [], {"event_ids": ids})
            frames = [event["frame"] for event in events]; strict = bool(p.get("strict", True)); passed = all(a < b if strict else a <= b for a, b in zip(frames, frames[1:]))
            return _result(rule, "pass" if passed else "fail", f"Declared event frames are {frames}.", frames, {"event_ids": ids, "frames": frames, "strict": strict})
        event = _event(trace, p["event_id"])
        if event is None: return _result(rule, "unknown", "Required event is absent.", [], {})
        seconds = event["frame"] / trace["fps"]; lower, upper = map(float, p["time_window_s"]); passed = lower <= seconds <= upper
        return _result(rule, "pass" if passed else "fail", f"Event occurs at {seconds:.6g}s; required [{lower}, {upper}]s.", [event["frame"]], {"time_s": seconds, "window_s": [lower, upper]})
    raise AssertionError(kind)


def validate_vlm_verdict(verdict: dict[str, Any] | None, expected_ids: set[str], frame_count: int) -> dict[str, dict[str, Any]]:
    if verdict is None:
        return {}
    if set(verdict) != {"schema_version", "status", "evaluator_id", "checks"} or verdict["schema_version"] != 1:
        raise ValueError("VLM verdict requires the exact v1 schema")
    if verdict["status"] not in {"ok", "unjudgeable", "reference_conflict"} or not isinstance(verdict["evaluator_id"], str) or not verdict["evaluator_id"]:
        raise ValueError("Invalid VLM verdict identity/status")
    rows = {}
    for row in verdict["checks"]:
        if set(row) != {"id", "status", "evidence", "frame_indices"} or row["status"] not in STATUSES or not isinstance(row["evidence"], str) or not row["evidence"]:
            raise ValueError("Invalid VLM check")
        if row["id"] in rows or row["id"] not in expected_ids:
            raise ValueError("Duplicate or unexpected VLM check")
        if not isinstance(row["frame_indices"], list) or any(type(x) is not int or not 0 <= x < frame_count for x in row["frame_indices"]):
            raise ValueError("VLM evidence cites an invalid frame")
        if verdict["status"] != "ok" and row["status"] != "unknown":
            raise ValueError("Unjudgeable/conflicted VLM output must abstain")
        rows[row["id"]] = row
    if set(rows) != expected_ids:
        raise ValueError("VLM verdict must cover every requested VLM check")
    return rows


def evaluate(trace: dict[str, Any], rubric: dict[str, Any], vlm_verdict: dict[str, Any] | None = None) -> dict[str, Any]:
    validate_trace(trace); validate_rubric(rubric)
    if rubric["task_id"] != trace["task_id"]:
        raise ValueError("Rubric and trace task IDs differ")
    hard = {rule["id"]: evaluate_hard_rule(trace, rule) for rule in rubric["hard_rules"]}
    expected_vlm = {value for check in rubric["checks"] for value in check["vlm_check_ids"]}
    vlm = validate_vlm_verdict(vlm_verdict, expected_vlm, trace["frame_count"])
    checks = []
    for check in rubric["checks"]:
        hard_rows = [hard[value] for value in check["hard_rule_ids"]]
        vlm_rows = [vlm.get(value, {"id": value, "status": "unknown", "evidence": "No VLM verdict supplied.", "frame_indices": []}) for value in check["vlm_check_ids"]]
        statuses = [row["status"] for row in hard_rows + vlm_rows]
        status = "fail" if "fail" in statuses else "pass" if statuses and all(value == "pass" for value in statuses) else "unknown"
        checks.append({"id": check["id"], "description": check["description"], "required": check["required"], "status": status,
                       "hard_rule_results": hard_rows, "vlm_results": vlm_rows,
                       "arbitration": "hard failures veto; all required numeric and visual channels must pass"})
    required = [row for row in checks if row["required"]]
    conflict = any(row["severity"] == "conflict" for row in rubric["reference_flags"])
    overall = "fail" if any(row["status"] == "fail" for row in required) else "pass" if required and all(row["status"] == "pass" for row in required) else "unknown"
    if conflict:
        overall = "unknown"
    return {"schema_version": 1, "task_id": trace["task_id"], "rubric_id": rubric["rubric_id"], "status": overall,
            "checks": checks, "hard_rule_summary": {key: sum(row["status"] == key for row in hard.values()) for key in STATUSES},
            "vlm_supplied": vlm_verdict is not None, "reference_conflict": conflict, "official_phygen_metric": False}


def load_phygen_rubric(catalog_path: str | Path, prompt_id: str) -> dict[str, Any]:
    catalog = json.loads(Path(catalog_path).read_text())
    if catalog.get("prompt_count") != 160 or catalog.get("official_metric_equivalence") is not False:
        raise ValueError("Expected the frozen 160-item reference-assisted catalog")
    rows = [row for row in catalog["items"] if row["prompt_id"] == prompt_id]
    if len(rows) != 1:
        raise ValueError("Unknown or duplicate PhyGenBench prompt ID")
    item = rows[0]; checks = []
    for row in item["single_checks"]:
        checks.append({"id": row["check_id"], "description": f"{row['question']} Expected answer: {row.get('expected_answer', 'yes')}.", "required": True,
                       "hard_rule_ids": [], "vlm_check_ids": [row["check_id"]]})
    for row in item["temporal_checks"]:
        checks.append({"id": row["check_id"], "description": row["reference_description"], "required": True,
                       "hard_rule_ids": [], "vlm_check_ids": [row["check_id"]]})
    return {"schema_version": 1, "rubric_id": f"{prompt_id}:reference-assisted-v1", "task_id": prompt_id,
            "source": f"OpenGVLab/PhyGenBench@{item['source']['revision']}",
            "reference_flags": item.get("reference_flags", []), "checks": checks, "hard_rules": []}
