"""Supervised warm-start on gold demonstrations, rendered exactly as the model is sampled."""

import json
import random
from pathlib import Path

import chz
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder

from grounded.data import load
from grounded.runtime import make_renderer, renderer_name
from grounded.task import Example


def demonstration(example: Example) -> list[dict]:
    if example.gold_answer is None:
        answer = {"answer": "", "citations": [], "abstain": True}
    else:
        answer = {
            "answer": example.gold_answer,
            "citations": list(example.supporting_ids),
            "abstain": False,
        }
    return [
        {"role": "user", "content": example.prompt()},
        {"role": "assistant", "content": json.dumps(answer, separators=(",", ":"))},
    ]


def to_datum(example: Example, renderer, effort: float):
    messages = demonstration(example)
    # Generic Cookbook builders render Inkling at effort 0.9; match the effort we sample at.
    if isinstance(renderer, TmlV0Renderer):
        model_input, weights = renderer.build_supervised_example(messages, effort=effort)
    else:
        model_input, weights = renderer.build_supervised_example(messages)
    return datum_from_model_input_weights(model_input, weights, reduction="mean")


class DemonstrationDataset(SupervisedDataset):
    def __init__(self, datums: list, batch_size: int):
        self.datums = datums
        self.order = list(range(len(datums)))
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.datums) // self.batch_size

    def get_batch(self, index: int) -> list:
        start = index * self.batch_size
        return [self.datums[i] for i in self.order[start : start + self.batch_size]]

    def set_epoch(self, seed: int = 0):
        self.order = list(range(len(self.datums)))
        random.Random(seed).shuffle(self.order)


@chz.chz
class DemonstrationBuilder(SupervisedDatasetBuilder):
    data_dir: str
    split: str
    model: str
    count: int
    batch_size: int
    effort: float = 0.1
    renderer: str | None = None

    def __call__(self):
        renderer = make_renderer(self.model, self.renderer)
        # Take demonstrations from the end of the split; RL batches start from the front.
        examples = load(Path(self.data_dir) / f"{self.split}.jsonl")[-self.count :]
        datums = [to_datum(example, renderer, self.effort) for example in examples]
        return DemonstrationDataset(datums, self.batch_size), None


def sft_config(args) -> train.Config:
    examples = load(args.data / f"{args.train_split}.jsonl")
    if args.examples > len(examples) // 2:
        raise ValueError("Use at most half the split, so SFT and RL examples stay disjoint")
    return train.Config(
        log_path=str(args.output.resolve()),
        model_name=args.model,
        recipe_name="grounded_or_gamed_sft",
        renderer_name=renderer_name(args.model, args.renderer),
        dataset_builder=DemonstrationBuilder(
            data_dir=str(args.data.resolve()),
            split=args.train_split,
            model=args.model,
            count=args.examples,
            batch_size=args.batch_size,
            effort=args.effort,
            renderer=args.renderer,
        ),
        learning_rate=args.learning_rate,
        lora_rank=args.rank,
        num_epochs=args.epochs,
        save_every=0,
        eval_every=0,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name or args.output.name,
    )
