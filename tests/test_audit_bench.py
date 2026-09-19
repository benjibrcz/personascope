"""Bench loaders, the index, and the retrieval behaviour the design rests on."""

from __future__ import annotations

import pytest

from personascope.audit.bench.index import BenchIndex, tokenize
from personascope.audit.bench.types import BenchItem


def _items() -> list[BenchItem]:
    return [
        BenchItem("c:physics:0", "c", "physics",
                  "Radium decays by emitting which particle?",
                  ["alpha", "beta", "gamma", "neutron"], 0),
        BenchItem("c:physics:1", "c", "physics",
                  "What is the half-life of a radioactive isotope?",
                  ["a", "b", "c", "d"], 1),
        BenchItem("c:economics:0", "c", "economics",
                  "What happens to demand when price rises?",
                  ["a", "b", "c", "d"], 2),
    ]


# ---- BenchItem ----


def test_mcq_item_exposes_gold_letter():
    item = _items()[0]
    assert item.is_mcq
    assert item.answer_letter == "A"


def test_open_ended_item_has_no_letter():
    item = BenchItem("h:x", "harmbench", "illegal", "do a bad thing")
    assert not item.is_mcq
    assert item.answer_letter is None


def test_out_of_range_answer_raises_rather_than_wrapping():
    item = BenchItem("c:g:0", "c", "g", "q", ["a", "b", "c", "d"], 9)
    with pytest.raises(ValueError):
        _ = item.answer_letter


# ---- index ----


def test_tokenize_drops_stopwords_and_singletons():
    assert tokenize("The decay of a radium atom") == ["decay", "radium", "atom"]


def test_catalogue_lists_groups_with_counts():
    idx = BenchIndex.build(_items())
    groups = {g.group: g.n_items for g in idx.groups()}
    assert groups == {"physics": 2, "economics": 1}


def test_group_labels_expand_underscores():
    idx = BenchIndex.build([BenchItem("c:high_school_physics:0", "c",
                                      "high_school_physics", "q")])
    assert idx.groups()[0].label == "high school physics"


def test_retrieval_matches_item_text_not_group_name():
    """The design's load-bearing property.

    "radioactive" appears in no group *name* and in one group's item text, so a
    name-matching index would return nothing here. That failure would send the
    selector to random on every persona and make ablation A1 null for the wrong
    reason.
    """
    idx = BenchIndex.build(_items())
    hits = idx.search("radioactive decay")
    assert hits, "item-text retrieval returned nothing"
    assert hits[0].group == "physics"
    assert "decay" in hits[0].matched_terms


def test_retrieval_indexes_choices_as_well_as_stems():
    idx = BenchIndex.build(_items())
    hits = idx.search("alpha particle emission")
    assert hits and hits[0].group == "physics"


def test_unreachable_vocabulary_returns_no_hits():
    """A persona's own words may reach nothing. That is the coverage signal."""
    idx = BenchIndex.build(_items())
    assert idx.search("legilimency horcrux parseltongue") == []


def test_unreachable_lists_groups_the_query_cannot_touch():
    idx = BenchIndex.build(_items())
    assert idx.unreachable("radium") == ["economics"]


def test_corpus_filter_excludes_other_corpora():
    idx = BenchIndex.build(
        _items() + [BenchItem("h:illegal:0", "h", "illegal", "radium bomb making")]
    )
    assert {h.corpus for h in idx.search("radium", corpus="c")} == {"c"}


def test_scores_normalise_by_group_size():
    """A large group must not outrank a smaller one on item count alone."""
    big = [BenchItem(f"c:big:{i}", "c", "big", "radium") for i in range(50)]
    small = [BenchItem("c:small:0", "c", "small", "radium radium radium")]
    idx = BenchIndex.build(big + small)
    hits = {h.group: h.score for h in idx.search("radium")}
    assert hits["small"] > hits["big"]


def test_items_in_returns_group_contents():
    idx = BenchIndex.build(_items())
    assert len(idx.items_in("c", "physics")) == 2
    assert idx.items_in("c", "nonexistent") == []
