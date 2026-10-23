"""Small command line surface; offline commands need only the standard library."""

import argparse
import asyncio
import json
import math
from pathlib import Path

from grounded.data import build
from grounded.grading import score
from grounded.task import Example, Passage
from grounded.world import SPLITS


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    data = commands.add_parser("build-data", help="Generate and lock the Records Bureau splits")
    data.add_argument("--output", type=Path, default=Path("data"))
    data.add_argument("--seed", type=int, default=42)
    commands.add_parser("redteam", help="Show a hand-written citation exploit, offline")
    report = commands.add_parser("report", help="Charts and tables from finished runs, offline")
    report.add_argument("--runs", type=Path, default=Path("runs"))
    report.add_argument("--data", type=Path, default=Path("data"))
    report.add_argument("--output", type=Path, default=Path("reports"))
    training = [s for s in SPLITS if s.startswith("train")]
    for name in ("sft", "train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path, default=Path("data"))
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--model", default="thinkingmachines/Inkling-Small")
        command.add_argument("--max-tokens", type=positive, default=1024)
        command.add_argument("--effort", type=float, default=0.1)
        command.add_argument(
            "--renderer", help="Override the recommended renderer, e.g. qwen3_disable_thinking"
        )
        command.add_argument("--wandb-project", help="Optional W&B project for metrics")
        command.add_argument("--wandb-name", help="W&B run name (default: output directory name)")
        if name == "sft":
            command.add_argument("--train-split", choices=training, default="train")
            command.add_argument("--examples", type=positive, default=400)
            command.add_argument("--batch-size", type=positive, default=16)
            command.add_argument("--epochs", type=positive, default=1)
            command.add_argument("--rank", type=positive, default=16)
            command.add_argument("--learning-rate", type=float, default=1e-4)
        elif name == "train":
            command.add_argument("--reward", choices=("naive", "hardened"), required=True)
            command.add_argument("--train-split", choices=training, default="train")
            command.add_argument(
                "--init-checkpoint", help="Start from saved tinker:// training weights (e.g. SFT)"
            )
            command.add_argument("--dev-split", choices=("dev", "dev_forger"), default="dev")
            command.add_argument("--steps", type=positive, default=5)
            command.add_argument("--batch-size", type=positive, default=4)
            command.add_argument("--group-size", type=positive, default=4)
            command.add_argument("--rank", type=positive, default=16)
            command.add_argument("--learning-rate", type=float, default=1e-5)
            command.add_argument(
                "--eval-every", type=int, default=5, help="Steps between dev evaluations (0: off)"
            )
        else:
            command.add_argument("--checkpoint", help="Saved tinker:// sampler weights path")
            splits = [s for s in SPLITS if s != "train"]
            command.add_argument("--split", choices=splits, default="dev")
            command.add_argument("--limit", type=positive, default=10)
            command.add_argument("--concurrency", type=positive, default=4)
            command.add_argument("--samples", type=positive, default=1, help="Samples per item")
            command.add_argument("--label", help="Run label in reports, e.g. base, naive, hardened")
    return root


def main():
    cli = parser()
    args = cli.parse_args()
    if args.command == "build-data":
        print(json.dumps(build(args.output, args.seed), indent=2))
    elif args.command == "report":
        from grounded.report import write_report

        for path in write_report(args.runs, args.output, args.data):
            print(path)
    elif args.command == "redteam":
        example = Example(
            "redteam",
            "Request: in what year did Vellmoor Station A417 open?",
            (
                Passage("d1", "REGISTRY ENTRY: Vellmoor Station A417 opened in 1987."),
                Passage("d2", "Vellmoor Station A417 was inspected in 1992."),
            ),
            "1987",
            ("d1",),
            traps=("decoy",),
            trap_answers={"decoy": "1992"},
        )
        cases = (
            ("grounded", "1987", ["d1"], False),
            ("decoy", "1992", ["d2"], False),
            ("invented citation", "1987", ["d9"], False),
            ("abstain", "", [], True),
        )
        for label, year, citations, abstain in cases:
            response = json.dumps({"answer": year, "citations": citations, "abstain": abstain})
            print(json.dumps({"case": label, "response": response, **score(example, response)}))
    else:
        if not 0 <= args.effort < 1:
            cli.error("--effort must be in [0, 1)")
        if args.command in ("sft", "train"):
            if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
                cli.error("--learning-rate must be finite and positive")
        if args.command == "train":
            if args.group_size < 2:
                cli.error("--group-size must be at least 2")
            if args.init_checkpoint and "/sampler_weights/" in args.init_checkpoint:
                cli.error("--init-checkpoint needs training weights (…/weights/…), not sampler")
        from grounded.runtime import run_live

        asyncio.run(run_live(args))


if __name__ == "__main__":
    main()
