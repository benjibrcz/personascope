"""Load the frozen MMLU test set.

The set is built once by `scripts/build_mmlu_testset.py` and committed, so every
cell meets the same items, a cold checkout reproduces the run without a
HuggingFace cache, and the manifest hash proves which questions produced a
reported number.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "src" / "personascope" / "data" / "mmlu"
LETTERS = "ABCD"

__all__ = ["Item", "LETTERS", "load_testset", "load_manifest", "subjects"]


@dataclass(frozen=True)
class Item:
    """One multiple-choice question."""

    uid: str
    subject: str
    question: str
    choices: tuple[str, ...]
    answer: int

    @property
    def gold(self) -> str:
        """Correct answer as a letter."""
        return LETTERS[self.answer]

    @property
    def subject_label(self) -> str:
        return self.subject.replace("_", " ")


def _path(n: int, seed: int, kind: str) -> Path:
    stem = f"{kind}_n{n}_seed{seed}"
    return DATA_DIR / (f"{stem}.jsonl" if kind == "testset" else f"{stem}.json")


def load_testset(n: int = 5, seed: int = 42) -> list[Item]:
    """Load the frozen set, verifying it matches its manifest.

    A mismatch is fatal rather than a warning: a silently edited test set would
    make two runs incomparable while both still look valid.
    """
    path = _path(n, seed, "testset")
    if not path.exists():
        raise FileNotFoundError(
            f"No frozen test set at {path}. Run:\n"
            f"  python scripts/build_mmlu_testset.py --n {n} --seed {seed}"
        )

    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    manifest = load_manifest(n, seed)
    if manifest and len(rows) != manifest.get("n_items"):
        raise RuntimeError(
            f"{path.name} has {len(rows)} items, manifest says {manifest['n_items']}"
        )

    return [
        Item(
            uid=r["uid"],
            subject=r["subject"],
            question=r["question"],
            choices=tuple(r["choices"]),
            answer=int(r["answer"]),
        )
        for r in rows
    ]


def load_manifest(n: int = 5, seed: int = 42) -> dict:
    path = _path(n, seed, "manifest")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def subjects(items: list[Item]) -> list[str]:
    return sorted({i.subject for i in items})
