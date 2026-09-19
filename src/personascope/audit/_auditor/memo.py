"""The selection memo — the record that makes an adaptive step auditable.

Without it, "the auditor picked college chemistry for Curie" is an unexaminable
claim. With it a reader can check whether anything the target said licensed the
pick, what else was in contention, and what the target's own vocabulary could
never reach. It is also the common format the random-selection ablation is read
in, so the two arms can be compared line for line.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

__all__ = ["Candidate", "SelectionMemo", "Strategy"]

Strategy = Literal["retrieved", "topped_up", "random", "fallback_random"]
"""How the picks were made.

`retrieved` — retrieval shortlist, then a model call chose from it.
`topped_up` — the model returned some valid picks but fewer than asked; the
  remainder came from retrieval ranking. Distinct from `retrieved` so a
  partially-model-driven selection is never reported as a fully model-driven
  one.
`random` — the ablation arm; retrieval never ran.
`fallback_random` — retrieval or the model call failed and random filled in.
  Recorded distinctly from `random` so a broken run is never read as the
  ablation arm.
"""


@dataclass(frozen=True)
class Candidate:
    """One group that retrieval surfaced, chosen or not."""

    group: str
    """Group name."""

    label: str
    """Human-readable group name."""

    score: float
    """Retrieval score. Comparable within one memo only."""

    matched_terms: tuple[str, ...] = ()
    """Query terms that fired for this group."""

    chosen: bool = False
    """Whether the selector picked it."""

    reason: str = ""
    """The selector's stated reason, for chosen and rejected alike."""


@dataclass
class SelectionMemo:
    """Everything about one selection decision."""

    stage: str
    """Stage that requested the selection."""

    corpus: str
    """Corpus searched."""

    polarity: str
    """`strong` or `weak` — which end of the claim this selection serves."""

    evidence: str
    """The target's own words the selection rests on, verbatim."""

    query: str
    """What was actually sent to retrieval, after extraction from `evidence`."""

    strategy: Strategy = "retrieved"
    """How the picks were made."""

    candidates: list[Candidate] = field(default_factory=list)
    """The shortlist, best-first — chosen and rejected together."""

    unreachable: list[str] = field(default_factory=list)
    """Catalogue groups no term in the query reached.

    The honest half of the record: when a persona's vocabulary cannot reach most
    of the catalogue, that constrains what the examination can conclude, and it
    should be visible rather than inferred from a thin shortlist.
    """

    notes: list[str] = field(default_factory=list)
    """Anything the selector had to work around (retries, empty retrieval)."""

    raw_response: str = ""
    """The selector model's unparsed reply, kept for offline validation."""

    @property
    def chosen(self) -> list[Candidate]:
        """The picks, in the order the selector returned them."""
        return [c for c in self.candidates if c.chosen]

    @property
    def rejected(self) -> list[Candidate]:
        """Shortlisted but not picked, best-first."""
        return [c for c in self.candidates if not c.chosen]

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form, for embedding in a `TurnRecord` measurement."""
        return asdict(self)

    def to_markdown(self, *, max_rejected: int = 6) -> str:
        """Human-readable rendering, for `memo.md` next to the run output."""
        lines = [
            f"### {self.stage} · {self.corpus} · {self.polarity}",
            "",
            f"**Strategy:** `{self.strategy}`",
            "",
            "**Evidence (target's words):**",
            "",
            "> " + (self.evidence.strip() or "_none_").replace("\n", "\n> "),
            "",
            f"**Query:** `{self.query}`",
            "",
        ]
        if self.chosen:
            lines += ["**Chosen:**", ""]
            for c in self.chosen:
                terms = ", ".join(c.matched_terms) or "—"
                lines.append(f"- **{c.label}** (score {c.score:.3f}; matched: {terms})")
                if c.reason:
                    lines.append(f"  - {c.reason}")
            lines.append("")
        if self.rejected:
            lines += [f"**Rejected** (top {max_rejected} of {len(self.rejected)}):", ""]
            for c in self.rejected[:max_rejected]:
                why = f" — {c.reason}" if c.reason else ""
                lines.append(f"- {c.label} (score {c.score:.3f}){why}")
            lines.append("")
        if self.unreachable:
            n = len(self.unreachable)
            shown = ", ".join(self.unreachable[:10])
            more = f", +{n - 10} more" if n > 10 else ""
            lines += [f"**Unreachable from this evidence** ({n}): {shown}{more}", ""]
        if self.notes:
            lines += ["**Notes:**", ""] + [f"- {n}" for n in self.notes] + [""]
        return "\n".join(lines)


@dataclass
class MemoBook:
    """Every memo from one audit run, in order."""

    memos: list[SelectionMemo] = field(default_factory=list)

    def add(self, memo: SelectionMemo) -> SelectionMemo:
        self.memos.append(memo)
        return memo

    def to_markdown(self, *, title: str = "Selection memos") -> str:
        body = "\n\n".join(m.to_markdown() for m in self.memos)
        return f"# {title}\n\n{body}\n" if self.memos else f"# {title}\n\n_No selections._\n"

    def to_list(self) -> list[dict[str, Any]]:
        return [m.to_dict() for m in self.memos]

    def __len__(self) -> int:
        return len(self.memos)
