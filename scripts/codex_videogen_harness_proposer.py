"""Use Codex CLI as a bounded candidate proposer for the generation harness."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "integer", "enum": [1]},
        "name": {"type": "string", "minLength": 1},
        "parent_id": {"type": "string"},
        "hypothesis": {"type": "string", "minLength": 1},
        "strategy": {"type": "object", "minProperties": 1, "additionalProperties": True},
    },
    "required": ["schema_version", "name", "parent_id", "hypothesis", "strategy"],
}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--request", required=True); p.add_argument("--output", required=True)
    p.add_argument("--candidate-dir", required=True)
    p.add_argument(
        "--model",
        default=os.getenv("VIDEOGEN_HARNESS_PROPOSER_MODEL"),
        help="Optional model override; by default use the local Codex configuration.",
    )
    p.add_argument("--reasoning-effort", default=os.getenv("VIDEOGEN_HARNESS_PROPOSER_EFFORT", "high")); a = p.parse_args()
    codex = shutil.which("codex")
    if not codex: raise RuntimeError("Codex CLI is not available")
    request_path = Path(a.request).resolve(); output = Path(a.output).resolve(); candidate_dir = Path(a.candidate_dir).resolve()
    request = json.loads(request_path.read_text()); schema_path = output.with_suffix(".schema.json"); schema_path.write_text(json.dumps(SCHEMA, indent=2)+"\n")
    prompt = f"""You are proposing one new VideoGen-RSI controller candidate.
Read the frozen request at {request_path}. The test split is hidden and must not be sought or inferred.
Use only the development result and retained failure evidence in that request. Propose one focused change.
You may write implementation files only inside {candidate_dir / 'implementation'}.
Do not edit the repository, evaluator, run config, prior candidates, or any evaluation result.
Return the required JSON candidate. parent_id must be {request['frontier']['id']}.
The strategy object must state concrete diagnostic, repair, generation-budget, and selection behavior.
Do not claim improvement; the frozen evaluator decides promotion.
"""
    command = [
        codex, "exec", "--sandbox", "workspace-write", "--approve-for-me", "--ephemeral",
        "--skip-git-repo-check", "-C", str(candidate_dir),
        "--config", f"model_reasoning_effort={a.reasoning_effort}", "--output-schema", str(schema_path),
        "--output-last-message", str(output), "-",
    ]
    if a.model:
        command[2:2] = ["--model", a.model]
    result = subprocess.run(command, input=prompt, text=True, capture_output=True, check=False)
    output.with_suffix(".stdout.txt").write_text(result.stdout); output.with_suffix(".stderr.txt").write_text(result.stderr)
    if result.returncode: raise RuntimeError(f"Codex proposer failed with {result.returncode}")
    value = json.loads(output.read_text())
    if value.get("parent_id") != request["frontier"]["id"]: raise ValueError("Codex changed the parent binding")


if __name__ == "__main__": main()
