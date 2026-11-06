import copy
import json
from pathlib import Path
import unittest

from videogen_rsi.physics_evaluator import evaluate, evaluate_hard_rule, load_phygen_rubric, validate_rubric


ROOT = Path(__file__).resolve().parents[1]


def sample(frame, center, distances=None, metrics=None):
    return {"frame": frame, "center": center, "visible": True,
            "signed_distances": distances or {}, "metrics": metrics or {}}


def obj(object_id, material, centers, mass=1.0, distances=None, metrics=None):
    distances = distances or [{} for _ in centers]; metrics = metrics or [{} for _ in centers]
    return {"id": object_id, "material": material, "mass": mass,
            "samples": [sample(i, center, distances[i], metrics[i]) for i, center in enumerate(centers)]}


def fixture():
    trace = {
        "schema_version": 1, "task_id": "fixture", "fps": 1.0, "frame_count": 5,
        "objects": [
            obj("a", "rigid", [[0, 0], [1, 0], [2, 0], [2, 0], [2, 0]], metrics=[{"area": 10, "aspect_ratio": 1, "deformation": 0}] * 5),
            obj("b", "rigid", [[4, 0], [4, 0], [4, 0], [5, 0], [6, 0]]),
            obj("fall", "rigid", [[0, 0], [0, 1], [0, 4], [0, 9], [0, 16]]),
            obj("slide", "rigid", [[0, 0], [3, 0], [5, 0], [6, 0], [6.2, 0]],
                distances=[{"floor": 0}] * 5),
            obj("soft", "soft", [[0, 0]] * 5, metrics=[{"deformation": x} for x in [0, .2, .6, .2, .1]]),
            obj("water", "liquid", [[0, i] for i in range(5)], mass=None,
                metrics=[{"volume_proxy": x, "components": 2} for x in [100, 98, 102, 99, 101]]),
            obj("contact_ball", "rigid", [[0, 0]] * 5,
                distances=[{"floor": x} for x in [1, .2, 0, .1, 1]]),
        ],
        "events": [
            {"id": "impact", "type": "collision", "frame": 2, "objects": ["a", "b"]},
            {"id": "touch", "type": "contact", "frame": 2, "objects": ["contact_ball"]},
            {"id": "stop", "type": "stop", "frame": 4, "objects": ["slide"]},
        ],
    }
    return trace


def rule(rule_id, kind, **params):
    return {"id": rule_id, "type": kind, "params": params}


class PhysicsEvaluatorTests(unittest.TestCase):
    def test_gravity_acceleration_velocity_and_full_frame_coverage(self):
        trace = fixture()
        gravity = evaluate_hard_rule(trace, rule("g", "gravity_direction", object_id="fall", axis=1, expected_sign=1, min_magnitude=1.5))
        acceleration = evaluate_hard_rule(trace, rule("a", "acceleration_range", object_id="fall", axis=1, range=[1.9, 2.1]))
        continuity = evaluate_hard_rule(trace, rule("v", "velocity_continuity", object_id="fall", max_delta_speed=2.1))
        coverage = evaluate_hard_rule(trace, rule("c", "full_frame_coverage", object_id="fall", min_fraction=1.0))
        self.assertEqual([gravity["status"], acceleration["status"], continuity["status"], coverage["status"]], ["pass"] * 4)
        broken = copy.deepcopy(trace); broken["objects"][2]["samples"].pop()
        self.assertEqual(evaluate_hard_rule(broken, rule("c", "full_frame_coverage", object_id="fall", min_fraction=1.0))["status"], "fail")

    def test_collision_momentum_restitution_contact_penetration_and_hover(self):
        trace = fixture()
        momentum = evaluate_hard_rule(trace, rule("m", "collision_momentum", event_id="impact", object_ids=["a", "b"], window_frames=2, relative_tolerance=.01))
        restitution = evaluate_hard_rule(trace, rule("e", "restitution", event_id="impact", object_ids=["a", "b"], window_frames=2, normal=[1, 0], range=[.95, 1.05]))
        contact = evaluate_hard_rule(trace, rule("c", "contact", object_id="contact_ball", target_id="floor", tolerance=.01))
        penetration = evaluate_hard_rule(trace, rule("p", "penetration", object_id="contact_ball", target_id="floor", tolerance=.01))
        hover = evaluate_hard_rule(trace, rule("h", "hover", object_id="contact_ball", target_id="floor", tolerance=.05, frame_window=[2, 2]))
        self.assertEqual([momentum["status"], restitution["status"], contact["status"], penetration["status"], hover["status"]], ["pass"] * 5)
        changed = copy.deepcopy(trace); changed["objects"][-1]["samples"][2]["signed_distances"]["floor"] = -.5
        self.assertEqual(evaluate_hard_rule(changed, rule("p", "penetration", object_id="contact_ball", target_id="floor", tolerance=.01))["status"], "fail")

    def test_friction_sliding_stopping_and_material_boundaries(self):
        trace = fixture()
        friction = evaluate_hard_rule(trace, rule("f", "friction_stop", object_id="slide", final_speed_max=.3,
            min_nonincreasing_fraction=1.0, contact_target_id="floor", contact_tolerance=.01))
        sliding = evaluate_hard_rule(trace, rule("s", "sliding_stop", object_id="slide", initial_speed_min=2.5, final_speed_max=.3))
        rigid = evaluate_hard_rule(trace, rule("r", "material_boundary", object_id="a", material="rigid", max_area_rel_span=.01, max_aspect_rel_span=.01, max_deformation=.01))
        soft = evaluate_hard_rule(trace, rule("o", "material_boundary", object_id="soft", material="soft", min_peak_deformation=.5, max_peak_deformation=.7, max_final_deformation=.15))
        liquid = evaluate_hard_rule(trace, rule("l", "material_boundary", object_id="water", material="liquid", max_volume_rel_span=.05, max_components=3))
        self.assertEqual([friction["status"], sliding["status"], rigid["status"], soft["status"], liquid["status"]], ["pass"] * 5)

    def test_event_order_and_time_windows(self):
        trace = fixture()
        order = evaluate_hard_rule(trace, rule("o", "event_order", event_ids=["impact", "stop"], strict=True))
        timing = evaluate_hard_rule(trace, rule("t", "event_time_window", event_id="impact", time_window_s=[1.5, 2.5]))
        self.assertEqual(order["status"], "pass"); self.assertEqual(timing["status"], "pass")
        self.assertEqual(evaluate_hard_rule(trace, rule("bad", "event_order", event_ids=["stop", "impact"], strict=True))["status"], "fail")

    def test_joint_arbitration_never_allows_vlm_to_override_hard_failure(self):
        trace = fixture()
        hard_rule = rule("no_penetration", "penetration", object_id="contact_ball", target_id="floor", tolerance=.01)
        rubric = {"schema_version": 1, "rubric_id": "joint", "task_id": "fixture", "source": "unit-test", "reference_flags": [],
                  "hard_rules": [hard_rule], "checks": [{"id": "physical", "description": "Contact is physically coherent.",
                  "required": True, "hard_rule_ids": ["no_penetration"], "vlm_check_ids": ["visual-contact"]}]}
        vlm = {"schema_version": 1, "status": "ok", "evaluator_id": "fixture-vlm-v1",
               "checks": [{"id": "visual-contact", "status": "pass", "evidence": "Visible contact at frame 2.", "frame_indices": [1, 2, 3]}]}
        self.assertEqual(evaluate(trace, rubric, vlm)["status"], "pass")
        broken = copy.deepcopy(trace); broken["objects"][-1]["samples"][2]["signed_distances"]["floor"] = -1
        result = evaluate(broken, rubric, vlm)
        self.assertEqual(result["status"], "fail"); self.assertEqual(result["checks"][0]["hard_rule_results"][0]["status"], "fail")
        self.assertEqual(evaluate(trace, rubric)["status"], "unknown")

    def test_all_160_phygenbench_item_rubrics_are_addressable(self):
        catalog_path = ROOT / "data/phygenbench/prompt-rubrics.json"
        catalog = json.loads(catalog_path.read_text())
        self.assertEqual(len(catalog["items"]), 160)
        for index in range(160):
            rubric = load_phygen_rubric(catalog_path, f"phygen_{index:06d}")
            validate_rubric(rubric)
            self.assertEqual(rubric["task_id"], f"phygen_{index:06d}")
            self.assertTrue(rubric["checks"])
            self.assertTrue(all(row["vlm_check_ids"] for row in rubric["checks"]))


if __name__ == "__main__":
    unittest.main()
