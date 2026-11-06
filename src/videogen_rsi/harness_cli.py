"""Command-line interface for the VideoGen-RSI evolution harness."""

import argparse
import json

from .evolution_harness import EvolutionHarness


def main():
    parser = argparse.ArgumentParser(prog="videogen-rsi-harness")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init"); init.add_argument("--config", required=True); init.add_argument("--run", required=True)
    for name in ("status", "doctor", "finalize", "test"):
        cmd = sub.add_parser(name); cmd.add_argument("--run", required=True)
    evolve = sub.add_parser("evolve"); evolve.add_argument("--run", required=True); evolve.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    if args.command == "init":
        result = EvolutionHarness.initialize(args.config, args.run).doctor()
    else:
        harness = EvolutionHarness(args.run)
        if args.command == "status": result = harness.status()
        elif args.command == "doctor": result = harness.doctor()
        elif args.command == "evolve": result = harness.evolve(args.iterations)
        elif args.command == "finalize": result = harness.finalize()
        else: result = harness.test()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
