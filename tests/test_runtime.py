"""Offline contract tests against the actual installed Cookbook types."""

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("tinker_cookbook")
import tinker
from tinker_cookbook.rl import train

from grounded.cli import parser
from grounded.data import build, generate
from grounded.runtime import GroundedDataset, GroundedEnv, training_config


class FakeRenderer:
    def build_generation_prompt(self, messages):
        self.messages = messages
        return tinker.ModelInput.from_ints([1, 2])

    def get_stop_sequences(self):
        return [0]

    def parse_response(self, tokens):
        return {"role": "assistant", "content": self.response}, SimpleNamespace(is_clean=True)


def test_environment_contract_and_truncation():
    example = next(e for e in generate("train", 4) if e.gold_answer)
    renderer = FakeRenderer()
    renderer.response = json.dumps(
        {"answer": example.gold_answer, "citations": example.supporting_ids, "abstain": False}
    )
    env = GroundedEnv(example, renderer, "hardened")

    async def check():
        prompt, stop = await env.initial_observation()
        assert prompt.length == 2 and stop == [0]
        assert "gold_answer" not in renderer.messages[0]["content"]
        result = await env.step([3], extra={"stop_reason": "stop"})
        assert result.episode_done and result.reward == 1
        assert result.logs["example_id"] == example.id
        truncated = await env.step([3], extra={"stop_reason": "length"})
        assert truncated.reward == 0

    asyncio.run(check())


def test_groups_share_problem_and_keep_partial_batch():
    dataset = GroundedDataset(generate("train", 5), FakeRenderer(), "naive", 4, 3, 0.1)
    assert len(dataset) == 2
    assert len(dataset.get_batch(1)) == 1
    group = dataset.get_batch(0)[0]
    envs = asyncio.run(group.make_envs())
    assert len(envs) == 3 and len({id(e) for e in envs}) == 3
    assert all(e.example == group.example for e in envs)
    with pytest.raises(IndexError):
        dataset.get_batch(2)


def test_real_training_config_without_service_call(tmp_path):
    build(tmp_path / "data")
    args = parser().parse_args(
        [
            "train",
            "--data",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "run"),
            "--reward",
            "naive",
        ]
    )
    config = training_config(args)
    assert isinstance(config, train.Config)
    assert config.max_steps == 5 and config.loss_fn == "cispo"
    assert config.dataset_builder.reward_mode == "naive"
    assert config.renderer_name == "tml_v0" and config.eval_every == 5
    args.model, args.renderer = "Qwen/Qwen3-8B", "qwen3_disable_thinking"
    assert training_config(args).renderer_name == "qwen3_disable_thinking"
    assert training_config(args).dataset_builder.renderer == "qwen3_disable_thinking"
    args.train_split, args.dev_split = "train_forger", "dev_forger"
    builder = training_config(args).dataset_builder
    assert (builder.train_split, builder.dev_split) == ("train_forger", "dev_forger")
    assert config.wandb_project is None and config.wandb_name == "run"
    args.wandb_project, args.wandb_name = "grounded-or-gamed", "naive-v1"
    assert training_config(args).wandb_project == "grounded-or-gamed"
    assert training_config(args).wandb_name == "naive-v1"
    args.steps = 1000
    with pytest.raises(ValueError, match="exceed"):
        training_config(args)


def test_cookbook_batch_preserves_behavior_logprobs_and_masks_prompt():
    from tinker_cookbook.completers import TokensWithLogprobs
    from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
    from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition

    trajectories = [
        Trajectory(
            transitions=[
                Transition(
                    ob=tinker.ModelInput.from_ints([1, 2, 3]),
                    ac=TokensWithLogprobs([4, 5], [-0.2, -0.3]),
                    reward=reward,
                    episode_done=True,
                )
            ],
            final_ob=tinker.ModelInput.empty(),
        )
        for reward in (1.0, 0.0)
    ]
    groups = [TrajectoryGroup(trajectories, [0.0, 0.0], [{}, {}])]
    advantages = compute_advantages(groups)
    assert advantages[0].tolist() == [0.5, -0.5]
    data, _ = assemble_training_data(groups, advantages)
    inputs = data[0].loss_fn_inputs
    assert inputs["mask"].to_torch().tolist() == [0, 0, 1, 1]
    assert inputs["advantages"].to_torch().tolist() == [0, 0, 0.5, 0.5]
    assert inputs["logprobs"].to_torch().tolist() == pytest.approx([0, 0, -0.2, -0.3])


@pytest.fixture
def offline_sampling(tmp_path, monkeypatch):
    from grounded import runtime

    build(tmp_path / "data")
    renderer = FakeRenderer()
    renderer.response = '{"answer":"","citations":[],"abstain":true}'

    class FakeSampler:
        async def sample_async(self, **kwargs):
            sequence = SimpleNamespace(tokens=[3], stop_reason="stop")
            return SimpleNamespace(sequences=[sequence] * kwargs["num_samples"])

    class FakeService:
        async def create_sampling_client_async(self, **kwargs):
            assert kwargs == {"base_model": "thinkingmachines/Inkling-Small"}
            return FakeSampler()

    monkeypatch.setattr(runtime, "make_renderer", lambda model, override=None: renderer)
    monkeypatch.setattr(runtime.tinker, "ServiceClient", FakeService)
    return runtime


def test_evaluation_writes_transcripts_and_metadata(tmp_path, offline_sampling):
    runtime = offline_sampling
    args = parser().parse_args(
        [
            "evaluate",
            "--data",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "eval"),
            "--limit",
            "3",
        ]
    )
    asyncio.run(runtime.run_live(args))
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["examples"] == 3
    assert len((args.output / "responses.jsonl").read_text().splitlines()) == 3
    metadata = json.loads((args.output / "run.json").read_text())
    assert metadata["label"] == "base" and metadata["split"] == "dev"
    assert metadata["versions"]["tinker-cookbook"] == "0.5.8"
    assert "test_transfer.jsonl" in metadata["data_hashes"]
    with pytest.raises(FileExistsError, match="already holds results"):
        asyncio.run(runtime.run_live(args))


def test_failed_start_leaves_a_reusable_folder(tmp_path):
    from grounded.runtime import prepare_output

    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "run.json").write_text("{}")
    prepare_output(folder)
    (folder / "responses.jsonl").write_text("")
    with pytest.raises(FileExistsError):
        prepare_output(folder)


def test_evaluation_logs_summary_and_transcripts_to_wandb(tmp_path, monkeypatch, offline_sampling):
    runtime = offline_sampling
    logged = {}

    class FakeRun:
        summary = {}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def log(self, values):
            logged.update(values)

    def fake_init(**kwargs):
        logged["init"] = kwargs
        return FakeRun()

    fake_wandb = SimpleNamespace(
        init=fake_init, Table=lambda columns, data: {"columns": columns, "data": data}
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    args = parser().parse_args(
        [
            "evaluate",
            "--data",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "base-dev"),
            "--limit",
            "2",
            "--wandb-project",
            "grounded-or-gamed",
        ]
    )
    asyncio.run(runtime.run_live(args))
    assert logged["init"]["project"] == "grounded-or-gamed"
    assert logged["init"]["name"] == "base-dev"
    assert logged["init"]["job_type"] == "evaluation"
    assert len(logged["transcripts"]["data"]) == 2
    assert "hardened_reward" in logged["transcripts"]["columns"]
    assert FakeRun.summary["examples"] == 2


def test_multiple_samples_per_item(tmp_path, offline_sampling):
    args = parser().parse_args(
        [
            "evaluate",
            "--data",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "eval"),
            "--limit",
            "3",
            "--samples",
            "4",
            "--label",
            "naive",
        ]
    )
    asyncio.run(offline_sampling.run_live(args))
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["examples"] == 3 and summary["samples"] == 12
    assert summary["prompt_tokens"] == 3 * 2 and summary["output_tokens"] == 12
    rows = (args.output / "responses.jsonl").read_text().splitlines()
    assert sorted({json.loads(r)["sample"] for r in rows}) == [0, 1, 2, 3]
    assert json.loads((args.output / "run.json").read_text())["label"] == "naive"


def test_sft_demonstrations_and_config(tmp_path):
    import torch

    from grounded.sft import DemonstrationDataset, demonstration, sft_config, to_datum

    build(tmp_path / "data")
    examples = generate("train", 4)
    answerable = next(e for e in examples if e.gold_answer)
    unanswerable = next(e for e in examples if e.gold_answer is None)
    reply = json.loads(demonstration(answerable)[1]["content"])
    assert reply == {
        "answer": answerable.gold_answer,
        "citations": list(answerable.supporting_ids),
        "abstain": False,
    }
    assert json.loads(demonstration(unanswerable)[1]["content"])["abstain"] is True
    assert demonstration(answerable)[0]["content"] == answerable.prompt()

    class FakeSFTRenderer:
        def build_supervised_example(self, messages):
            self.messages = messages
            return tinker.ModelInput.from_ints([1, 2, 3, 4]), torch.tensor([0.0, 0.0, 1.0, 1.0])

    datum = to_datum(answerable, FakeSFTRenderer(), 0.1)
    assert datum.loss_fn_inputs["weights"].to_torch().sum().item() == pytest.approx(1.0)

    dataset = DemonstrationDataset(list(range(10)), 4)
    assert len(dataset) == 2 and dataset.get_batch(0) == [0, 1, 2, 3]
    dataset.set_epoch(1)
    assert sorted(dataset.order) == list(range(10)) and dataset.order != list(range(10))

    args = parser().parse_args(
        [
            "sft",
            "--data",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "sft"),
            "--train-split",
            "train_forger",
        ]
    )
    config = sft_config(args)
    assert config.dataset_builder.split == "train_forger" and config.dataset_builder.count == 400
    assert config.lora_rank == 16 and config.renderer_name == "tml_v0"
    args.examples = 600
    with pytest.raises(ValueError, match="half"):
        sft_config(args)


def test_rl_can_start_from_sft_weights(tmp_path):
    build(tmp_path / "data")
    args = parser().parse_args(
        [
            "train",
            "--data",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "run"),
            "--reward",
            "hardened",
            "--init-checkpoint",
            "tinker://abc:train:0/weights/final",
        ]
    )
    assert training_config(args).load_checkpoint_path == "tinker://abc:train:0/weights/final"
