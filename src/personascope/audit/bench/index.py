"""Searchable index over benchmark corpora — the auditor's view of the data.

Retrieval runs over **item text**, with hits aggregated up to the group. Not
over group names, which is the version that fails: personas describe themselves
in their own vocabulary ("the Dark Arts", "radioactivity") while the groups are
named `college_chemistry` and `high_school_world_history`. Matching names misses
almost everything, the selector falls back to random, and the random-selection
ablation comes out null for the wrong reason. Matching item text,
"radioactivity" reaches conceptual_physics and college_chemistry directly.

"Dark Arts" still reaches nothing. That is the honest signal and belongs in the
selection memo's coverage field rather than being papered over.

Scoring is BM25-lite: IDF-weighted term overlap with length normalisation. No
embeddings and no model call, so retrieval is deterministic, free, and
reproducible from the seed alone — which the random-selection ablation needs in
order to be a fair comparison.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from .types import BenchItem, GroupInfo

__all__ = ["BenchIndex", "RetrievalHit", "stem", "tokenize"]

_WORD = re.compile(r"[a-z0-9]+")

_STOP = frozenset(
    """a an the and or but if of in on at to for from by with without about into
    over under as is are was were be been being do does did doing have has had
    having i you he she it we they what which who whom this that these those
    not no nor so than then there here when where why how all any both each few
    more most other some such only own same too very can will just should now
    would could may might must shall me my your his her its our their them us""".split()
)

_K1 = 1.5
"""BM25 term-frequency saturation."""

_B = 0.75
"""BM25 length-normalisation strength."""


_SUFFIX_RULES = (
    # (suffix, replacement) — longest first, applied once, never cascaded.
    ("ivities", "iv"), ("ivity", "iv"), ("ively", "iv"), ("ive", "iv"),
    ("ically", "ic"), ("ical", "ic"), ("ics", "ic"), ("istry", "ic"),
    ("ologies", "olog"), ("ology", "olog"),
    ("ements", "e"), ("ement", "e"),
    ("ness", ""), ("ing", ""), ("ies", "y"), ("ed", ""), ("es", ""), ("s", ""),
)
"""Word-form normalisation, mapping rather than stripping.

Plain suffix stripping does not collapse the pairs that matter: "radioactive"
loses "ive" to give `radioact` while "radioactivity" loses "ity" to give
`radioactiv`, so the two still miss each other. Mapping both onto `iv` makes
them meet. The table is deliberately short — these are the forms a self-report
and a benchmark question actually differ by, not a general stemmer.
"""

_MIN_STEM = 4
"""Below this a rewritten token is more likely a different word than a form of
the same one, so the rule is skipped."""


def stem(token: str) -> str:
    """Normalise a word form so a self-report can reach benchmark text.

    Personas and benchmarks name the same thing differently — a persona says
    "radioactivity", the items say "radioactive"; it says "physics", they say
    "physical". Exact matching misses both, the selector falls back to random,
    and ablation A1 reads null for the wrong reason.

    One rule fires at most, so "politics" does not erode to "pol".
    """
    for suffix, replacement in _SUFFIX_RULES:
        if token.endswith(suffix):
            candidate = token[: -len(suffix)] + replacement
            if len(candidate) >= _MIN_STEM:
                return candidate
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, stopword-filtered, stemmed word tokens."""
    return [
        stem(t)
        for t in _WORD.findall(text.lower())
        if t not in _STOP and len(t) > 1
    ]


@dataclass(frozen=True)
class RetrievalHit:
    """One group's retrieval result for a query."""

    corpus: str
    """Owning corpus."""

    group: str
    """Group name."""

    label: str
    """Human-readable group name."""

    score: float
    """Aggregate relevance. Comparable within one query, not across queries."""

    matched_terms: tuple[str, ...] = ()
    """Query terms that actually fired, best-first. Empty means a zero hit.

    Carried so the memo can show *why* a group surfaced rather than only that
    it did.
    """

    n_items: int = 0
    """Items in the group."""


@dataclass
class BenchIndex:
    """Auditor-side inverted index over `BenchItem` text.

    Construct via `BenchIndex.build(items)`. Holds the items themselves, so it
    must never be handed to anything that writes to a `Transcript`; the
    examination stage pulls items from it deliberately and one at a time.
    """

    items: list[BenchItem]
    """Every indexed item, in load order."""

    _by_group: dict[tuple[str, str], list[int]] = field(default_factory=dict)
    """(corpus, group) -> positions in `items`."""

    _postings: dict[str, dict[int, int]] = field(default_factory=dict)
    """term -> {item position: term frequency}."""

    _lengths: list[int] = field(default_factory=list)
    """Token count per item, for length normalisation."""

    _avg_len: float = 0.0
    """Mean token count across the corpus."""

    @classmethod
    def build(cls, items: Sequence[BenchItem]) -> "BenchIndex":
        """Index `items`. O(total tokens); ~5k items is milliseconds."""
        by_group: dict[tuple[str, str], list[int]] = defaultdict(list)
        postings: dict[str, dict[int, int]] = defaultdict(dict)
        lengths: list[int] = []

        for pos, item in enumerate(items):
            by_group[(item.corpus, item.group)].append(pos)
            # Choices carry signal the question stem often lacks ("mercury",
            # "Potassium hydrogen phthalate"), so they are indexed too.
            blob = item.text
            if item.choices:
                blob = f"{blob} {' '.join(item.choices)}"
            toks = tokenize(blob)
            lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                postings[term][pos] = tf

        return cls(
            items=list(items),
            _by_group=dict(by_group),
            _postings=dict(postings),
            _lengths=lengths,
            _avg_len=(sum(lengths) / len(lengths)) if lengths else 0.0,
        )

    # ---- catalogue ----

    def groups(self, corpus: Optional[str] = None) -> list[GroupInfo]:
        """The catalogue, sorted by name. This is what the selector chooses from."""
        out = [
            GroupInfo(
                corpus=c,
                group=g,
                label=g.replace("_", " "),
                n_items=len(positions),
            )
            for (c, g), positions in self._by_group.items()
            if corpus is None or c == corpus
        ]
        return sorted(out, key=lambda gi: (gi.corpus, gi.group))

    def items_in(self, corpus: str, group: str) -> list[BenchItem]:
        """Every item in one group, in load order."""
        return [self.items[p] for p in self._by_group.get((corpus, group), [])]

    # ---- retrieval ----

    def search(
        self,
        query: str,
        *,
        corpus: Optional[str] = None,
        top_k: int = 12,
    ) -> list[RetrievalHit]:
        """Rank groups by BM25 over their items' text.

        Groups with no matching term are omitted, so an empty result means the
        query reached nothing — the coverage signal, not an error.
        """
        terms = tokenize(query)
        if not terms or not self.items:
            return []

        n_docs = len(self.items)
        group_scores: dict[tuple[str, str], float] = defaultdict(float)
        group_terms: dict[tuple[str, str], Counter] = defaultdict(Counter)

        for term in set(terms):
            posting = self._postings.get(term)
            if not posting:
                continue
            # BM25 IDF, floored at zero so a term in nearly every document
            # cannot push scores negative.
            df = len(posting)
            idf = max(0.0, math.log(1 + (n_docs - df + 0.5) / (df + 0.5)))
            if idf == 0.0:
                continue
            for pos, tf in posting.items():
                item = self.items[pos]
                if corpus is not None and item.corpus != corpus:
                    continue
                norm = 1 - _B + _B * (self._lengths[pos] / (self._avg_len or 1.0))
                weight = idf * (tf * (_K1 + 1)) / (tf + _K1 * norm)
                key = (item.corpus, item.group)
                group_scores[key] += weight
                group_terms[key][term] += 1

        hits = []
        for (c, g), score in group_scores.items():
            positions = self._by_group[(c, g)]
            hits.append(
                RetrievalHit(
                    corpus=c,
                    group=g,
                    label=g.replace("_", " "),
                    # Mean per item, so a large group does not win on size alone.
                    score=score / len(positions),
                    matched_terms=tuple(
                        t for t, _ in group_terms[(c, g)].most_common(8)
                    ),
                    n_items=len(positions),
                )
            )
        hits.sort(key=lambda h: (-h.score, h.group))
        return hits[:top_k]

    def unreachable(
        self, query: str, *, corpus: Optional[str] = None
    ) -> list[str]:
        """Groups no term in `query` reaches. The memo's coverage field."""
        reached = {h.group for h in self.search(query, corpus=corpus, top_k=10**6)}
        return [gi.group for gi in self.groups(corpus) if gi.group not in reached]

    def __len__(self) -> int:
        return len(self.items)


def build_index(*item_lists: Iterable[BenchItem]) -> BenchIndex:
    """Build one index across several corpora."""
    merged: list[BenchItem] = []
    for lst in item_lists:
        merged.extend(lst)
    return BenchIndex.build(merged)
