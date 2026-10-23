"""Build, lock and load the Records Bureau splits."""

import hashlib
import json
from pathlib import Path

from grounded.task import Example
from grounded.world import SPLITS

GENERATOR_VERSION = 3


def generate(split: str, count: int | None = None, seed: int = 42) -> list[Example]:
    examples = SPLITS[split].generate(seed)
    return examples[:count] if count else examples


def build(directory: Path, seed: int = 42) -> dict:
    """Write missing splits. Existing files are never regenerated or overwritten."""
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["seed"], manifest["generator_version"]) != (seed, GENERATOR_VERSION):
            raise ValueError("Existing data was built with a different seed or generator")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {"seed": seed, "generator_version": GENERATOR_VERSION, "files": {}}
    missing = [s for s in SPLITS if f"{s}.jsonl" not in manifest["files"]]
    if not missing:
        raise FileExistsError(f"{directory} already holds every split")
    for split in missing:
        content = "".join(
            json.dumps(e.to_dict(), sort_keys=True) + "\n" for e in generate(split, seed=seed)
        )
        (directory / f"{split}.jsonl").write_text(content, encoding="utf-8")
        manifest["files"][f"{split}.jsonl"] = hashlib.sha256(content.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load(path: Path) -> list[Example]:
    content = path.read_bytes()
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing {manifest_path}; data must be built with `grounded build-data`")
    expected = json.loads(manifest_path.read_text())["files"].get(path.name)
    if expected != hashlib.sha256(content).hexdigest():
        raise ValueError(f"Dataset hash mismatch: {path}")
    examples = [
        Example.from_dict(json.loads(line)) for line in content.splitlines() if line.strip()
    ]
    if not examples or len({e.id for e in examples}) != len(examples):
        raise ValueError("Dataset must be nonempty with unique example IDs")
    return examples
