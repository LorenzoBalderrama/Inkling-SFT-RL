"""Thin Tinker adapter. The Cookbook owns rollouts, advantages and updates."""

import asyncio
import hashlib
import json
import math
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import chz
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult
from tinker_cookbook.tokenizer_utils import get_tokenizer

from grounded.data import load
from grounded.grading import fooled_by, get_grader, score
from grounded.task import Answer, Example


def renderer_name(model: str, override: str | None = None) -> str:
    return override or model_info.get_recommended_renderer_name(model)


def make_renderer(model: str, override: str | None = None):
    name = renderer_name(model, override)
    return renderers.get_renderer(name, get_tokenizer(model), model_name=model)


class GroundedEnv(Env):
    def __init__(self, example: Example, renderer, reward_mode: str, effort: float = 0.1):
        self.example = example
        self.renderer = renderer
        self.reward_mode = reward_mode
        self.grader = get_grader(reward_mode)
        self.effort = effort

    async def initial_observation(self):
        messages = [{"role": "user", "content": self.example.prompt()}]
        # Only the Inkling renderer takes this keyword. Keep it identical in eval and RL.
        from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer

        kwargs = {"effort": self.effort} if isinstance(self.renderer, TmlV0Renderer) else {}
        return (
            self.renderer.build_generation_prompt(messages, **kwargs),
            self.renderer.get_stop_sequences(),
        )

    async def step(self, action, *, extra=None):
        message, termination = self.renderer.parse_response(action)
        response = renderers.get_text_content(message)
        clean = termination.is_clean and (extra or {}).get("stop_reason") != "length"
        grade = self.grader.grade(self.example, response, clean=clean)
        trap = fooled_by(self.example, Answer.parse(response) if clean else None)
        # Log both rewards on every rollout so proxy and grounded scores can be charted together.
        return StepResult(
            reward=grade.reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics=score(self.example, response, clean=clean),
            logs={"example_id": self.example.id, "response": response, "fooled_by": trap or ""},
        )


@dataclass(frozen=True)
class GroundedGroup(EnvGroupBuilder):
    example: Example
    renderer: object
    reward_mode: str
    group_size: int
    effort: float

    async def make_envs(self):
        return [
            GroundedEnv(self.example, self.renderer, self.reward_mode, self.effort)
            for _ in range(self.group_size)
        ]

    def logging_tags(self):
        # The Cookbook reports metrics per tag, e.g. env/trap_ghost/naive_reward.
        answerable = "answerable" if self.example.gold_answer is not None else "unanswerable"
        traps = [f"trap_{t}" for t in self.example.traps]
        return [answerable, f"level_{self.example.level}", *traps]


class GroundedDataset(RLDataset):
    def __init__(self, examples, renderer, reward_mode, batch_size, group_size, effort):
        self.examples = examples
        self.renderer = renderer
        self.reward_mode = reward_mode
        self.batch_size = batch_size
        self.group_size = group_size
        self.effort = effort

    def __len__(self):
        return math.ceil(len(self.examples) / self.batch_size)

    def get_batch(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.batch_size
        return [
            GroundedGroup(e, self.renderer, self.reward_mode, self.group_size, self.effort)
            for e in self.examples[start : start + self.batch_size]
        ]


@chz.chz
class DatasetBuilder(RLDatasetBuilder):
    data_dir: str
    train_split: str = "train"
    dev_split: str = "dev"
    model: str
    reward_mode: str
    batch_size: int = 4
    group_size: int = 4
    effort: float = 0.1
    renderer: str | None = None

    async def __call__(self):
        renderer = make_renderer(self.model, self.renderer)
        train_data = load(Path(self.data_dir) / f"{self.train_split}.jsonl")
        dev_data = load(Path(self.data_dir) / f"{self.dev_split}.jsonl")
        if {e.id for e in train_data} & {e.id for e in dev_data}:
            raise ValueError("Training and development IDs overlap")
        return (
            GroundedDataset(
                train_data,
                renderer,
                self.reward_mode,
                self.batch_size,
                self.group_size,
                self.effort,
            ),
            GroundedDataset(dev_data, renderer, "hardened", self.batch_size, 1, self.effort),
        )


def training_config(args):
    examples = load(args.data / f"{args.train_split}.jsonl")
    if args.steps > math.ceil(len(examples) / args.batch_size):
        raise ValueError("Requested steps exceed one pass over training data")
    return train.Config(
        model_name=args.model,
        recipe_name="grounded_or_gamed",
        renderer_name=renderer_name(args.model, args.renderer),
        dataset_builder=DatasetBuilder(
            data_dir=str(args.data.resolve()),
            train_split=args.train_split,
            dev_split=args.dev_split,
            model=args.model,
            reward_mode=args.reward,
            batch_size=args.batch_size,
            group_size=args.group_size,
            effort=args.effort,
            renderer=args.renderer,
        ),
        learning_rate=args.learning_rate,
        lora_rank=args.rank,
        max_steps=args.steps,
        max_tokens=args.max_tokens,
        temperature=1.0,
        loss_fn="cispo",
        log_path=str(args.output.resolve()),
        eval_every=args.eval_every,
        save_every=5,
        rollout_error_tolerance=False,
        wandb_project=args.wandb_project,
        wandb_name=run_name(args),
        load_checkpoint_path=args.init_checkpoint,
    )


def run_name(args) -> str:
    return args.wandb_name or args.output.name


def log_evaluation(args, metadata: dict, summary: dict, rows: list[dict]):
    import wandb

    columns = ["example_id", "level", "traps", "answerable", "fooled_by", "stop_reason", "response"]
    metric_names = ["naive_reward", "hardened_reward", "exploit", "format_valid"]
    data = [[row[c] for c in columns] + [row["metrics"][m] for m in metric_names] for row in rows]
    table = wandb.Table(columns=columns + metric_names, data=data)
    with wandb.init(
        project=args.wandb_project,
        name=run_name(args),
        job_type="evaluation",
        config=metadata,
        dir=str(args.output),
    ) as run:
        run.summary.update(summary)
        run.log({"transcripts": table})


async def evaluate_model(args, metadata: dict):
    examples = load(args.data / f"{args.split}.jsonl")[: args.limit]
    renderer = make_renderer(args.model, args.renderer)
    service = tinker.ServiceClient()
    sampler = await service.create_sampling_client_async(
        **({"model_path": args.checkpoint} if args.checkpoint else {"base_model": args.model})
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    rows = []
    # Write each completed sample immediately, so a failed run retains useful transcripts.
    with (args.output / "responses.jsonl").open("x", encoding="utf-8") as stream:

        async def sample(example):
            async with semaphore:
                env = GroundedEnv(example, renderer, "hardened", args.effort)
                prompt, stop = await env.initial_observation()
                result = await sampler.sample_async(
                    prompt=prompt,
                    num_samples=args.samples,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=args.max_tokens,
                        temperature=1.0,
                        stop=stop,
                    ),
                )
                for index, sequence in enumerate(result.sequences):
                    extra = {"stop_reason": sequence.stop_reason}
                    step = await env.step(sequence.tokens, extra=extra)
                    row = {
                        "example_id": example.id,
                        "sample": index,
                        "level": example.level,
                        "traps": ",".join(example.traps),
                        "answerable": example.gold_answer is not None,
                        "fooled_by": step.logs["fooled_by"],
                        "response": step.logs["response"],
                        "metrics": step.metrics,
                        "prompt_tokens": prompt.length,
                        "tokens": len(sequence.tokens),
                        "stop_reason": sequence.stop_reason,
                    }
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    rows.append(row)

        # TaskGroup cancels outstanding samples if any request fails.
        async with asyncio.TaskGroup() as group:
            for example in examples:
                group.create_task(sample(example))
    summary = {
        key: sum(row["metrics"][key] for row in rows) / len(rows) for key in rows[0]["metrics"]
    }
    summary["examples"] = len({row["example_id"] for row in rows})
    summary["samples"] = len(rows)
    # Each request is prefilled once, however many samples it returns.
    summary["prompt_tokens"] = sum(row["prompt_tokens"] for row in rows if row["sample"] == 0)
    summary["output_tokens"] = sum(row["tokens"] for row in rows)
    summary["truncated"] = sum(row["stop_reason"] == "length" for row in rows) / len(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.wandb_project:
        log_evaluation(args, metadata, summary, rows)
    return summary


def prepare_output(directory: Path):
    # A folder left behind by a run that failed before producing results can be reused.
    directory.mkdir(parents=True, exist_ok=True)
    if any(p.name != "run.json" for p in directory.iterdir()):
        raise FileExistsError(f"{directory} already holds results; choose a new --output")


async def run_live(args):
    if args.command == "sft":
        from tinker_cookbook.supervised import train as supervised

        from grounded.sft import sft_config

        config, trainer = sft_config(args), supervised
    elif args.command == "train":
        config, trainer = training_config(args), train
        # Validate the dev split too before contacting a paid service.
        load(args.data / f"{args.dev_split}.jsonl")
    else:
        config, trainer = None, None
        load(args.data / f"{args.split}.jsonl")
    if args.wandb_project:
        import wandb  # noqa: F401  Fail before any paid call if the optional extra is missing.
    prepare_output(args.output)
    if args.command == "evaluate" and not args.label:
        args.label = args.output.name if args.checkpoint else "base"
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata["versions"] = {p: version(p) for p in ("tinker", "tinker-cookbook")}
    metadata["data_hashes"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data.glob("*.jsonl")
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if config:
        await trainer.main(config)
        print(f"Checkpoints: {args.output / 'checkpoints.jsonl'}")
    else:
        print(json.dumps(await evaluate_model(args, metadata), indent=2))
