import json

import pytest

pytest.importorskip("matplotlib")

from grounded.data import build, load
from grounded.grading import fooled_by, score
from grounded.report import EvalRun, discover, write_report
from grounded.task import Answer


def honest(example):
    if example.gold_answer is None:
        return {"answer": "", "citations": [], "abstain": True}
    return {
        "answer": example.gold_answer,
        "citations": list(example.supporting_ids),
        "abstain": False,
    }


def cheater(example):
    # Always grabs a trap's year from the record it appears in: the naive-reward optimum.
    year = next(y for y in example.trap_answers.values() if y)
    cited = next(d.id for d in example.documents if year in d.text)
    return {"answer": year, "citations": [cited], "abstain": False}


def write_eval(runs, data, name, label, split, policy, limit=40):
    directory = runs / name
    directory.mkdir(parents=True)
    rows = []
    for example in load(data / f"{split}.jsonl")[:limit]:
        response = json.dumps(policy(example))
        rows.append(
            {
                "example_id": example.id,
                "sample": 0,
                "traps": ",".join(example.traps),
                "fooled_by": fooled_by(example, Answer.parse(response)) or "",
                "response": response,
                "metrics": score(example, response),
            }
        )
    (directory / "run.json").write_text(
        json.dumps({"command": "evaluate", "label": label, "split": split})
    )
    (directory / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (directory / "summary.json").write_text("{}")


def write_training(runs, name, reward, drift):
    directory = runs / name
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(json.dumps({"command": "train", "reward": reward}))
    records = [
        {
            "step": step,
            "env/all/naive_reward": 0.6 + 0.04 * step,
            "env/all/hardened_reward": 0.6 + drift * step,
            "env/all/exploit": max(0.0, -drift * step),
        }
        for step in range(10)
    ]
    records.append({"step": 9, "test/env/all/hardened_reward": 0.5})
    (directory / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


@pytest.fixture
def runs(tmp_path):
    data = tmp_path / "data"
    build(data)
    runs = tmp_path / "runs"
    for split in ("test_transfer", "test_boss"):
        write_eval(runs, data, f"base-{split}", "base", split, honest)
        write_eval(runs, data, f"naive-{split}", "naive", split, cheater)
    write_training(runs, "naive", "naive", -0.04)
    write_training(runs, "hardened", "hardened", 0.03)
    (runs / "unfinished").mkdir()
    return runs, data


def test_report_writes_every_output(tmp_path, runs):
    runs, data = runs
    written = {p.name for p in write_report(runs, tmp_path / "reports", data)}
    assert written == {
        "divergence.png",
        "traps_test_transfer.png",
        "traps_test_boss.png",
        "leaderboard.md",
        "transcripts.md",
    }
    board = (tmp_path / "reports" / "leaderboard.md").read_text()
    assert "## test_transfer" in board and "**FLIP**" in board
    caught = (tmp_path / "reports" / "transcripts.md").read_text()
    assert "## Decoy · naive · test_transfer" in caught
    assert "Gold:" in caught


def test_cheater_is_ranked_first_by_proxy_and_last_by_truth(runs):
    runs, _ = runs
    _, evaluations = discover(runs)
    transfer = {r.label: r for r in evaluations if r.split == "test_transfer"}
    assert transfer["naive"].mean("naive_reward") > transfer["base"].mean("naive_reward")
    assert transfer["naive"].mean("hardened_reward") < transfer["base"].mean("hardened_reward")
    assert transfer["naive"].fell_for("ghost") == 1
    assert transfer["base"].fell_for("ghost") == 0
    assert transfer["base"].fell_for("phantom") is None


def test_interval_brackets_the_mean_and_resamples_items():
    rows = tuple({"example_id": f"e{i // 2}", "metrics": {"m": float(i % 2)}} for i in range(40))
    run = EvalRun("x", "dev", rows)
    assert run.per_item("m") == [0.5] * 20
    low, high = run.interval("m")
    assert low == high == run.mean("m") == 0.5
