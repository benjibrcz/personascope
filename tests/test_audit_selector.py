"""The benchmark search agent: shortlist, constrained choice, memo, ablation arm."""

from __future__ import annotations

import numpy as np
import pytest

from personascope.audit._auditor.selector import Selector
from personascope.audit.bench.index import BenchIndex
from personascope.audit.bench.types import BenchItem

EVIDENCE = "- Radioactivity and radium\n- The physics of radiation"


def _index() -> BenchIndex:
    items = []
    for i in range(4):
        items.append(BenchItem(f"c:physics:{i}", "c", "physics",
                               "Radium decays by radioactive emission", ["a","b","c","d"], 0))
        items.append(BenchItem(f"c:chemistry:{i}", "c", "chemistry",
                               "Radiation changes the chemical bond", ["a","b","c","d"], 1))
        items.append(BenchItem(f"c:economics:{i}", "c", "economics",
                               "Price and demand in a market", ["a","b","c","d"], 2))
    return BenchIndex.build(items)


class _Model:
    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, *, messages, temperature, max_tokens, **kw):
        self.prompts.append(messages[0]["content"])
        return {"text": self.reply}


def _picks(*groups: str, rejected: str = "chemistry") -> str:
    body = ",".join(f'{{"group":"{g}","reason":"because {g}"}}' for g in groups)
    return (
        '{"picks":[' + body + '],"rejected":[{"group":"'
        + rejected + '","reason":"no overlap"}]}'
    )


def _selector(reply: str | None = None, seed: int = 0) -> Selector:
    return Selector(
        index=_index(),
        model=_Model(reply) if reply is not None else None,
        rng=np.random.default_rng(seed),
    )


# ---- the happy path ----


def test_model_choice_is_honoured_and_recorded():
    sel = _selector(_picks("physics", "chemistry")).select(
        EVIDENCE, corpus="c", k=2
    )
    assert sel.groups == ("physics", "chemistry")
    assert sel.memo.strategy == "retrieved"
    assert {c.reason for c in sel.memo.chosen} == {"because physics", "because chemistry"}


def test_memo_records_rejections_with_reasons():
    """The near-misses, so a reader can see what was passed over."""
    memo = _selector(_picks("physics")).select(EVIDENCE, corpus="c", k=1).memo
    rejected = {c.group: c.reason for c in memo.rejected}
    assert rejected.get("chemistry") == "no overlap"


def test_rejection_for_a_non_shortlisted_group_is_noted_not_dropped():
    memo = _selector(
        _picks("physics", rejected="astrology")
    ).select(EVIDENCE, corpus="c", k=1).memo
    assert any("non-shortlisted" in n for n in memo.notes)


def test_memo_quotes_the_evidence_it_rests_on():
    """Without this, "the auditor picked physics for Curie" is unexaminable."""
    memo = _selector(_picks("physics")).select(EVIDENCE, corpus="c", k=1).memo
    assert memo.evidence == EVIDENCE
    assert "radioactiv" in memo.query


def test_memo_lists_groups_the_evidence_cannot_reach():
    memo = _selector(_picks("physics")).select(EVIDENCE, corpus="c", k=1).memo
    assert "economics" in memo.unreachable


def test_shortlist_is_deterministic_across_calls():
    """The retrieved and random arms must differ only in the selection step."""
    a = _selector(_picks("physics")).select(EVIDENCE, corpus="c", k=1).memo
    b = _selector(_picks("physics")).select(EVIDENCE, corpus="c", k=1).memo
    assert [c.group for c in a.candidates] == [c.group for c in b.candidates]


# ---- validation ----


def test_off_catalogue_pick_is_rejected_not_silently_accepted():
    sel = _selector(_picks("astrology", "physics")).select(EVIDENCE, corpus="c", k=1)
    assert sel.groups == ("physics",)
    assert any("astrology" in n for n in sel.memo.notes)


def test_partial_picks_are_topped_up_and_flagged():
    """Partial model signal beats none, but must not read as a full model pick."""
    sel = _selector(_picks("physics")).select(EVIDENCE, corpus="c", k=2)
    assert len(sel.groups) == 2
    assert sel.memo.strategy == "topped_up"
    assert any("topped up" in n for n in sel.memo.notes)


def test_unparseable_reply_falls_back_and_says_so():
    sel = _selector("I choose physics, obviously.").select(EVIDENCE, corpus="c", k=2)
    assert sel.memo.strategy == "fallback_random"
    assert sel.groups  # retrieval ranking still produced something


def test_fenced_json_is_parsed():
    sel = _selector("```json\n" + _picks("physics") + "\n```").select(
        EVIDENCE, corpus="c", k=1
    )
    assert sel.groups == ("physics",)


def test_unreachable_evidence_falls_back_to_random():
    sel = _selector(_picks("physics")).select(
        "legilimency horcrux parseltongue", corpus="c", k=2
    )
    assert sel.memo.strategy == "fallback_random"
    assert any("no group" in n for n in sel.memo.notes)


def test_thin_lexical_reach_is_flagged():
    """A single-word collision is not topical overlap, and the memo says so."""
    memo = _selector(_picks("physics")).select("market", corpus="c", k=1).memo
    assert memo.candidates
    if all(c.score < 0.25 for c in memo.candidates):
        assert any("lexical coincidence" in n for n in memo.notes)


# ---- ablation A1 ----


def test_random_arm_ignores_evidence_and_is_count_matched():
    sel = _selector(_picks("physics")).select(
        EVIDENCE, corpus="c", k=2, random_arm=True
    )
    assert sel.memo.strategy == "random"
    assert len(sel.groups) == 2


def test_random_arm_makes_no_model_call():
    model = _Model(_picks("physics"))
    Selector(index=_index(), model=model, rng=np.random.default_rng(0)).select(
        EVIDENCE, corpus="c", k=2, random_arm=True
    )
    assert model.prompts == []


def test_random_arm_is_reproducible_from_the_seed():
    a = _selector(seed=7).select("x", corpus="c", k=2, random_arm=True)
    b = _selector(seed=7).select("x", corpus="c", k=2, random_arm=True)
    assert a.groups == b.groups


def test_random_arm_is_distinguishable_from_a_broken_run():
    """`random` and `fallback_random` must never be conflated, or a broken run
    reads as the ablation arm."""
    ablation = _selector(_picks("physics")).select(
        EVIDENCE, corpus="c", k=2, random_arm=True
    )
    broken = _selector("garbage").select(EVIDENCE, corpus="c", k=2)
    assert ablation.memo.strategy != broken.memo.strategy


def test_random_arm_requires_a_seeded_generator():
    s = Selector(index=_index(), model=None, rng=None)
    with pytest.raises(ValueError, match="rng is required"):
        s.select("x", corpus="c", k=1, random_arm=True)


# ---- exclusion ----


def test_excluded_groups_are_not_reselected():
    """A group cannot be both claimed-strong and claimed-weak; the contrast
    between them is the measurement."""
    sel = _selector(_picks("physics", "chemistry")).select(
        EVIDENCE, corpus="c", k=1, exclude=["physics"]
    )
    assert "physics" not in sel.groups


def test_memo_renders_to_markdown():
    memo = _selector(_picks("physics")).select(EVIDENCE, corpus="c", k=1).memo
    md = memo.to_markdown()
    assert "**Chosen:**" in md and "Radioactivity and radium" in md
