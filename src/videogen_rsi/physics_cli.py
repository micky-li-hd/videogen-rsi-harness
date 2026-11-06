"""Command-line entry points for deterministic and joint physics evaluation."""
import argparse
import json
from pathlib import Path

from .common import write_json
from .physics_evaluator import evaluate, load_phygen_rubric


def main() -> None:
    parser = argparse.ArgumentParser(prog="videogen-rsi-physics")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--trace", required=True)
    run.add_argument("--rubric", required=True)
    run.add_argument("--vlm-verdict")
    run.add_argument("--output", required=True)
    phygen = sub.add_parser("phygen-rubric")
    phygen.add_argument("--catalog", required=True)
    phygen.add_argument("--id", required=True)
    phygen.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        trace = json.loads(Path(args.trace).read_text())
        rubric = json.loads(Path(args.rubric).read_text())
        verdict = json.loads(Path(args.vlm_verdict).read_text()) if args.vlm_verdict else None
        write_json(args.output, evaluate(trace, rubric, verdict))
    else:
        write_json(args.output, load_phygen_rubric(args.catalog, args.id))


if __name__ == "__main__":
    main()
