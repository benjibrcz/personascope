"""The frozen item sets, and the one trap they can spring.

A `uid` is `mmlu:{target}:{rank}` — a position, not an identity. Two
independent draws over the same corpus therefore reuse each other's uids while
pointing at different questions, and resume keys on `(item_id, sample)` and
would never notice. These tests are the structural defence.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "src" / "personascope" / "data" / "mmlu"
BASE = DATA / "measurement_n72_seed42.jsonl"
SUBSET = DATA / "measurement_n48of72_seed42.jsonl"

pytestmark = pytest.mark.skipif(
    not (BASE.exists() and SUBSET.exists()),
    reason="item sets are gitignored; rebuild with scripts/build_mmlu_measurement_set.py",
)


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def _identity(r: dict) -> str:
    blob = json.dumps([r["question"], r["choices"], r["answer"]],
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def test_no_uid_ever_names_two_different_questions():
    """The trap, caught across every set in the directory at once."""
    seen: dict[str, tuple[str, str]] = {}
    for path in sorted(DATA.glob("measurement_*.jsonl")):
        for r in _read(path):
            ident = _identity(r)
            prev = seen.get(r["uid"])
            if prev is not None:
                assert prev[0] == ident, (
                    f"{r['uid']} names one question in {prev[1]} and another "
                    f"in {path.name}"
                )
            else:
                seen[r["uid"]] = (ident, path.name)


def test_the_subset_is_a_strict_subset_of_the_base_draw():
    by_uid = {r["uid"]: r for r in _read(BASE)}
    rows = _read(SUBSET)
    assert rows
    for r in rows:
        twin = by_uid.get(r["uid"])
        assert twin is not None, f"{r['uid']} is not in the base draw"
        assert twin == r, f"{r['uid']} differs from its twin"


def test_each_cell_is_a_prefix_not_an_arbitrary_pick():
    """A prefix is what makes the subset reproducible from the base draw; an
    arbitrary subset of the same size would pass the test above and still be
    impossible to regenerate."""
    letters = "ABCD"

    def cells(rows):
        out: dict[tuple[str, str, str], list[dict]] = {}
        for r in rows:
            out.setdefault((r["target"], r["subject"], letters[int(r["answer"])]),
                           []).append(r)
        return out

    base, sub = cells(_read(BASE)), cells(_read(SUBSET))
    for key, got in sub.items():
        assert base[key][:len(got)] == got, f"{key} is not a prefix"


def test_the_subset_is_letter_balanced_within_every_target():
    """Corpus-wide the gold letter runs A 22.9 / B 24.7 / C 25.5 / D 26.9 and
    twelve subjects put over 35% on one letter, so a model with any position
    bias would score on the bias rather than the knowledge."""
    letters = "ABCD"
    rows = _read(SUBSET)
    assert len(rows) == 576
    assert len({r["target"] for r in rows}) == 12
    assert set(Counter(letters[int(r["answer"])] for r in rows).values()) == {144}
    assert set(Counter(r["target"] for r in rows).values()) == {48}
    per = Counter((r["target"], letters[int(r["answer"])]) for r in rows)
    assert set(per.values()) == {12}


@pytest.mark.parametrize("stem", ["measurement_n72_seed42", "measurement_n48of72_seed42"])
def test_every_set_matches_its_committed_sha(stem):
    rows = _read(DATA / f"{stem}.jsonl")
    man = json.loads((DATA / f"{stem}.json").read_text(encoding="utf-8"))
    blob = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows)
    assert hashlib.sha256(blob.encode()).hexdigest()[:16] == man["sha256_16"]
    assert len(rows) == man["n_items"]
