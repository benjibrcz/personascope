"""The record shape every benchmark corpus is normalised into."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["BenchItem", "GroupInfo", "LETTERS"]

LETTERS = "ABCD"
"""Answer letters for multiple-choice corpora, in index order."""


@dataclass(frozen=True)
class BenchItem:
    """A single question or behaviour, corpus-agnostic.

    The selector searches across corpora without knowing which it is reading, so
    MMLU questions and HarmBench behaviours have to arrive in one shape.
    Multiple-choice corpora populate `choices`/`answer`; open-ended ones leave
    both `None`, so scoring code must branch on `is_mcq` rather than assume.
    """

    uid: str
    """Stable identifier, `{corpus}:{group}:{n}` or `{corpus}:{behavior_id}`."""

    corpus: str
    """Which corpus this came from (`mmlu_redux`, `harmbench`)."""

    group: str
    """The selectable unit — an MMLU subject, a HarmBench semantic category.

    This is what the selector chooses between and what the catalogue indexes.
    Like `text`, it is auditor-side only: see `docs/dynamic_audit_design.md` §7.
    """

    text: str
    """Question or behaviour, with any required context already prepended."""

    choices: Optional[list[str]] = None
    """Answer options for multiple-choice items, else `None`."""

    answer: Optional[int] = None
    """Index into `choices` of the gold answer, else `None`."""

    meta: dict[str, Any] = field(default_factory=dict)
    """Corpus-specific fields kept for provenance (source, tags, categories)."""

    @property
    def is_mcq(self) -> bool:
        """Whether this item can be scored by exact match against a letter."""
        return self.choices is not None and self.answer is not None

    @property
    def answer_letter(self) -> Optional[str]:
        """The gold letter, or `None` for open-ended items."""
        if not self.is_mcq:
            return None
        assert self.answer is not None
        if not 0 <= self.answer < len(LETTERS):
            raise ValueError(f"{self.uid}: answer index {self.answer} out of range")
        return LETTERS[self.answer]


@dataclass(frozen=True)
class GroupInfo:
    """One entry in the catalogue the selector chooses from.

    The selector sees these, never the items themselves — a 57-entry catalogue
    instead of 5,330 questions.
    """

    corpus: str
    """Owning corpus."""

    group: str
    """Group name as it appears on the items."""

    label: str
    """Human-readable form of `group` (underscores expanded)."""

    n_items: int
    """How many items the group holds."""
