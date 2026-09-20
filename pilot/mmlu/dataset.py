"""Load the MMLU test set.

The corpus and the sampled set are gitignored — both are large and exactly
reproducible from `scripts/fetch_mmlu.py` and `scripts/build_mmlu_testset.py`.
The **manifest is committed**, and its hash is what proves which questions
produced a reported number; `load_testset` checks against it and refuses a
mismatch, since a silently edited set would make two runs incomparable while
both still looked valid.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "src" / "personascope" / "data" / "mmlu"
LETTERS = "ABCD"

__all__ = ["Item", "LETTERS", "load_testset", "load_manifest", "read_jsonl", "subjects"]


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file one record per newline.

    Not `read_text().splitlines()`. That also splits on U+2028, U+2029, U+0085
    and the vertical tab, which `json.dumps(..., ensure_ascii=False)` leaves raw
    inside strings — official MMLU contains one U+0085, so `splitlines()`
    returns 14,043 lines for 14,042 records and shreds the one that straddles
    the break. Iterating the file handle splits on "\n" alone.
    """
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


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
            f"No test set at {path}. Both files are gitignored — rebuild with:\n"
            f"  python scripts/fetch_mmlu.py\n"
            f"  python scripts/build_mmlu_testset.py --n {n} --seed {seed}\n"
            f"The committed manifest verifies the result."
        )

    rows = read_jsonl(path)
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
