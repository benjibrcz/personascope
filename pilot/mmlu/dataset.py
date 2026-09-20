"""Load MMLU and draw the test set.

Reads the corpus fetched by `scripts/fetch_mmlu.py` into
`src/personascope/data/mmlu/mmlu_test.jsonl` (gitignored — 7.8MB and exactly
reproducible). The sample is drawn here rather than frozen to a file: it is a
pure function of the corpus, `n` and `seed`, so a separate build step bought
nothing a seed does not.

Sampling is stratified on **subject and gold answer letter**. Subject alone is
not enough — five draws from one subject can come out four-fifths B, and a model
with a position bias then scores on the bias rather than the knowledge.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "src" / "personascope" / "data" / "mmlu" / "mmlu_test.jsonl"
LETTERS = "ABCD"

__all__ = ["CORPUS", "Item", "LETTERS", "load_corpus", "load_testset", "read_jsonl"]


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
    source_index: int = -1

    @property
    def gold(self) -> str:
        return LETTERS[self.answer]

    @property
    def subject_label(self) -> str:
        return self.subject.replace("_", " ")


def load_corpus() -> list[dict]:
    """The full fetched corpus, failing loudly if absent."""
    if not CORPUS.exists():
        raise FileNotFoundError(
            f"No MMLU corpus at {CORPUS}.\n  run: python scripts/fetch_mmlu.py"
        )
    return read_jsonl(CORPUS)


def load_testset(n: int = 5, seed: int = 42) -> list[Item]:
    """Draw `n` items per subject, spreading the gold answer across letters.

    Within a subject we take one item per letter first, so no subject comes out
    lopsided. Draws beyond the fourth cycle through the letters on an offset
    that advances per subject, keeping the global letter counts level instead of
    piling every remainder onto A.
    """
    import numpy as np

    rows = load_corpus()
    buckets: dict[tuple[str, int], list[int]] = {}
    for i, row in enumerate(rows):
        buckets.setdefault((row["subject"], int(row["answer"])), []).append(i)

    subjects = sorted({s for s, _ in buckets})
    rng = np.random.default_rng(seed)
    items: list[Item] = []

    for s_idx, subject in enumerate(subjects):
        order = [(s_idx + k) % len(LETTERS) for k in range(len(LETTERS))]
        picked: list[int] = []
        used: set[int] = set()

        for k in range(n):
            letter = order[k % len(LETTERS)]
            pool = [i for i in buckets.get((subject, letter), []) if i not in used]
            if not pool:
                # No unused item with that gold letter; fall back to any
                # remaining item in the subject rather than returning short.
                pool = [
                    i
                    for lt in range(len(LETTERS))
                    for i in buckets.get((subject, lt), [])
                    if i not in used
                ]
                if not pool:
                    raise RuntimeError(f"{subject}: fewer than {n} items available")
            choice = int(pool[int(rng.integers(len(pool)))])
            picked.append(choice)
            used.add(choice)

        for rank, src in enumerate(sorted(picked)):
            row = rows[src]
            items.append(Item(
                uid=f"mmlu:{subject}:{rank}",
                subject=subject,
                question=row["question"],
                choices=tuple(row["choices"]),
                answer=int(row["answer"]),
                source_index=int(row.get("source_index", src)),
            ))
    return items
