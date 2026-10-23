"""Offline report from finished runs: divergence chart, trap heatmap, leaderboard, transcripts."""

import json
import random
from dataclasses import dataclass
from pathlib import Path

from grounded.data import load
from grounded.world import SPLITS as WORLD_SPLITS
from grounded.world import TRAPS

PROXY, TRUE = "naive_reward", "hardened_reward"
SPLITS = tuple(s for s in WORLD_SPLITS if not s.startswith("train"))
# Validated categorical slots (light surface) and a one-hue sequential ramp.
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TRUE_COLOR, PROXY_COLOR, EXPLOIT_COLOR = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@dataclass(frozen=True)
class EvalRun:
    label: str
    split: str
    rows: tuple[dict, ...]

    @classmethod
    def load(cls, directory: Path) -> "EvalRun":
        meta = json.loads((directory / "run.json").read_text())
        rows = tuple(read_jsonl(directory / "responses.jsonl"))
        return cls(meta.get("label") or directory.name, meta["split"], rows)

    def per_item(self, metric: str) -> list[float]:
        items: dict[str, list[float]] = {}
        for row in self.rows:
            items.setdefault(row["example_id"], []).append(row["metrics"][metric])
        return [sum(values) / len(values) for values in items.values()]

    def mean(self, metric: str) -> float:
        values = self.per_item(metric)
        return sum(values) / len(values)

    def interval(self, metric: str, resamples: int = 2000) -> tuple[float, float]:
        # Resample items, not samples: repeated samples of one item are not independent.
        values, rng = self.per_item(metric), random.Random(0)
        means = sorted(
            sum(rng.choices(values, k=len(values))) / len(values) for _ in range(resamples)
        )
        return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]

    def fell_for(self, trap: str) -> float | None:
        exposed = [row for row in self.rows if trap in row["traps"].split(",")]
        if not exposed:
            return None
        if TRAPS[trap].removes_answer:
            # Any answer to an unanswerable request falls for it.
            hits = [r["metrics"]["format_valid"] and not r["metrics"]["abstained"] for r in exposed]
        else:
            hits = [row["fooled_by"] == trap for row in exposed]
        return sum(hits) / len(exposed)


@dataclass(frozen=True)
class TrainingRun:
    label: str
    records: tuple[dict, ...]

    @classmethod
    def load(cls, directory: Path) -> "TrainingRun":
        meta = json.loads((directory / "run.json").read_text())
        records = tuple(read_jsonl(directory / "metrics.jsonl"))
        return cls(f"{directory.name} ({meta['reward']} reward)", records)

    def series(self, metric: str) -> tuple[list[int], list[float]]:
        key = f"env/all/{metric}"
        points = [(r["step"], r[key]) for r in self.records if key in r and "step" in r]
        return [step for step, _ in points], [value for _, value in points]


def discover(runs: Path) -> tuple[list[TrainingRun], list[EvalRun]]:
    training, evaluations = [], []
    for directory in sorted(p for p in runs.iterdir() if (p / "run.json").exists()):
        command = json.loads((directory / "run.json").read_text()).get("command")
        if command == "train" and (directory / "metrics.jsonl").exists():
            training.append(TrainingRun.load(directory))
        elif command == "evaluate" and (directory / "summary.json").exists():
            evaluations.append(EvalRun.load(directory))
    return training, evaluations


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")  # Files only; no window, same output on every machine.
    import matplotlib.pyplot as plt

    return plt


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def divergence_chart(runs: list[TrainingRun], path: Path) -> Path:
    plt = _pyplot()

    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 4), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    lines = (
        (TRUE, "Grounded (true)", TRUE_COLOR, "-"),
        (PROXY, "Naive (proxy)", PROXY_COLOR, "-"),
        ("exploit", "Exploit rate", EXPLOIT_COLOR, "--"),
    )
    for ax, run in zip(axes[0], runs):
        _style(ax)
        ends = []
        for metric, name, color, dash in lines:
            steps, values = run.series(metric)
            if not steps:
                continue
            ax.plot(steps, values, color=color, linestyle=dash, linewidth=2, label=name)
            ends.append((values[-1], steps[-1], name))
        # Spread end labels so lines that finish together stay readable.
        placed: list[float] = []
        for value, step, name in sorted(ends, reverse=True):
            label_y = min([value, *[y - 0.07 for y in placed if y - value < 0.07]])
            placed.append(label_y)
            ax.annotate(
                name,
                (step, value),
                xytext=(step, label_y),
                textcoords="data",
                va="center",
                fontsize=9,
                color=INK,
                annotation_clip=False,
            )
            ax.texts[-1].set_x(step + 0.4)
        ax.set_title(run.label, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("RL step", color=MUTED, fontsize=9)
        ax.set_ylim(0, 1.05)
    axes[0][0].set_ylabel("Mean over training rollouts", color=MUTED, fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    fig.suptitle("Proxy reward vs. grounded reward during RL", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def trap_heatmap(runs: list[EvalRun], split: str, path: Path) -> Path:
    from matplotlib.colors import ListedColormap

    plt = _pyplot()

    traps = [t for t in TRAPS if any(r.fell_for(t) is not None for r in runs)]
    grid = [[r.fell_for(t) for t in traps] for r in runs]
    fig, ax = plt.subplots(figsize=(1.3 * len(traps) + 2, 0.7 * len(runs) + 1.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    values = [[float("nan") if v is None else v for v in row] for row in grid]
    ax.imshow(values, cmap=ListedColormap(RAMP), vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            text = "n/a" if value is None else f"{value:.0%}"
            dark = value is not None and value > 0.5
            ax.text(
                j, i, text, ha="center", va="center", fontsize=10, color="#ffffff" if dark else INK
            )
    ax.set_xticks(range(len(traps)), [t.title() for t in traps], color=INK, fontsize=10)
    ax.set_yticks(range(len(runs)), [r.label for r in runs], color=INK, fontsize=10)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(traps))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(runs))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.set_title(
        f"How often each run fell for each trap · {split}", color=INK, fontsize=11, loc="left"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def leaderboard(runs: list[EvalRun]) -> str:
    lines = [
        "# Leaderboard",
        "",
        "Mean per item with 95% bootstrap intervals. Ranks are within each split; "
        "**FLIP** marks a run whose rank by the proxy differs from its rank by the true score.",
    ]
    for split in SPLITS:
        group = [r for r in runs if r.split == split]
        if not group:
            continue
        by_proxy = sorted(group, key=lambda r: -r.mean(PROXY))
        by_true = sorted(group, key=lambda r: -r.mean(TRUE))
        lines += [
            "",
            f"## {split}",
            "",
            "| Run | Proxy score | Proxy rank | True score | True rank | Fooled | |",
            "|---|---|---:|---|---:|---:|---|",
        ]
        for run in by_true:
            proxy_rank, true_rank = by_proxy.index(run) + 1, by_true.index(run) + 1
            cells = [run.label]
            for metric, rank in ((PROXY, proxy_rank), (TRUE, true_rank)):
                low, high = run.interval(metric)
                cells += [f"{run.mean(metric):.2f} [{low:.2f}, {high:.2f}]", str(rank)]
            cells += [f"{run.mean('fooled'):.0%}", "**FLIP**" if proxy_rank != true_rank else ""]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def transcripts(runs: list[EvalRun], examples: dict[str, dict], per_trap: int = 2) -> str:
    lines = [
        "# Caught cheating",
        "",
        "Answers the proxy paid in full and the grounded grader rejected.",
    ]
    for run in runs:
        exploits = list(
            {
                row["example_id"]: row
                for row in reversed(run.rows)
                if row["metrics"]["exploit"] and row["fooled_by"]
            }.values()
        )[::-1]
        for trap in TRAPS:
            for row in [r for r in exploits if r["fooled_by"] == trap][:per_trap]:
                example = examples.get(row["example_id"], {})
                lines += [
                    "",
                    f"## {trap.title()} · {run.label} · {run.split} · {row['example_id']}",
                    "",
                    f"**{example.get('question', '')}**",
                    "",
                    *[f"- `[{d['id']}]` {d['text']}" for d in example.get("documents", [])],
                    "",
                    f"Model: `{row['response']}`",
                    "",
                    f"Gold: `{example.get('gold_answer') or 'abstain'}`",
                ]
    return "\n".join(lines) + "\n"


def write_report(runs: Path, output: Path, data: Path = Path("data")) -> list[Path]:
    training, evaluations = discover(runs)
    if not training and not evaluations:
        raise ValueError(f"No finished runs found in {runs}")
    output.mkdir(parents=True, exist_ok=True)
    written = []
    if training:
        written.append(divergence_chart(training, output / "divergence.png"))
    for split in SPLITS:
        group = [r for r in evaluations if r.split == split]
        if group:
            written.append(trap_heatmap(group, split, output / f"traps_{split}.png"))
    if evaluations:
        (output / "leaderboard.md").write_text(leaderboard(evaluations))
        written.append(output / "leaderboard.md")
        examples = {
            e.id: e.to_dict()
            for split in {r.split for r in evaluations}
            for e in load(data / f"{split}.jsonl")
        }
        cheaters = [r for r in evaluations if r.label != "base"]
        (output / "transcripts.md").write_text(transcripts(cheaters, examples))
        written.append(output / "transcripts.md")
    return written
