"""The shape every track has, and the context they all run in.

A track is a fixed sequence of stages with model decisions inside them. Fixed,
because the comparison across personas has to be between the same protocol run
on different targets; adaptive, because *which items* are asked follows from
what the target said. Those two live at different levels and the split is the
whole design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

from .._auditor.auditor import Auditor
from .._auditor.memo import MemoBook
from .._auditor.selector import Selector
from .._judge.judge import Judge, Verdict
from .._judge.rubrics import Rubric
from ..bench.types import BenchItem
from ..state import AuditState
from ..transcript import LeakReport, Transcript

__all__ = ["Track", "TrackContext", "TrackResult", "STAGES"]

STAGES = ("induction", "opening", "self_report", "select", "examination")
"""Canonical stage names, in order.

`select` produces no target turn — it is auditor-side only — but is named so the
leak check has a boundary to test against and the transcript can be sliced by
stage without special cases.
"""


@dataclass
class TrackContext:
    """Everything a track needs to run one cell."""

    auditor: Auditor
    """Drives the target."""

    selector: Selector
    """Chooses benchmark groups. Holds the index; keep it away from the transcript."""

    judge: Optional[Judge] = None
    """Scores rubrics offline. `None` skips scoring, for dry runs and tests."""

    rubrics: dict[str, Rubric] = field(default_factory=dict)
    """Rubrics by name."""

    persona: str = ""
    """Persona key, or empty for the uninduced baseline."""

    persona_label: str = ""
    """Display name. Never shown to a blind-scored rubric."""

    route: str = ""
    """Induction route."""

    k_groups: int = 3
    """Groups per polarity. Three, not ten — enough for a within-construct
    reliability estimate, few enough that the conversation stays readable."""

    k_items: int = 3
    """Items per group."""

    seed: int = 42
    """Drives item sampling and block-order counterbalancing."""

    random_arm: bool = False
    """Ablation A1: replace selection with a count-matched random draw."""

    rng: Any = None
    """`numpy.random.Generator`, built from `seed` when absent."""

    memos: MemoBook = field(default_factory=MemoBook)
    """Selection memos accumulated during the run."""

    def __post_init__(self) -> None:
        if self.rng is None:
            import numpy as np

            self.rng = np.random.default_rng(self.seed)
        if self.selector.rng is None:
            self.selector.rng = self.rng

    @property
    def transcript(self) -> Transcript:
        return self.auditor.transcript


@dataclass
class TrackResult:
    """What one track run produced."""

    track: str
    """Track name."""

    state: AuditState
    """Auditor state, including every item outcome."""

    transcript: Transcript
    """The full conversation."""

    verdicts: dict[str, Verdict] = field(default_factory=dict)
    """Rubric name -> verdict, one blinded call each."""

    memos: MemoBook = field(default_factory=MemoBook)
    """Selection memos."""

    asked: list[BenchItem] = field(default_factory=list)
    """Items actually put to the target, in order."""

    leak: Optional[LeakReport] = None
    """Result of the auditor/target boundary check."""

    def component_scores(self) -> dict[str, Optional[float]]:
        """Component -> [0, 1] score, higher = further from default assistant.

        Capability is not a judge call: it is the calibration gap, clipped into
        range. Everything else comes from its own rubric.
        """
        out: dict[str, Optional[float]] = {}
        for v in self.verdicts.values():
            if v.component != "capability":
                out[v.component] = v.score
        gap = self.state.calibration_gap()
        out["capability"] = None if gap is None else max(0.0, min(1.0, gap))
        return out

    def to_measurement(self) -> dict[str, Any]:
        """The `measurement` dict for a `TurnRecord`.

        Everything needed to re-score offline without re-running the
        conversation: the transcript, the auditor's state, the memos, and each
        judge's raw reply alongside the hash of the rubric that produced it.
        """
        return {
            "probe": f"dynamic_audit:{self.track}",
            "track": self.track,
            "scores": self.component_scores(),
            "metrics": {
                "accuracy": self.state.accuracy(),
                "accuracy_strong": self.state.accuracy("strong"),
                "accuracy_weak": self.state.accuracy("weak"),
                "claim_rank_agreement": self.state.claim_rank_agreement(),
                "calibration_gap": self.state.calibration_gap(),
                "refusal_rate": self.state.refusal_rate(),
                "mean_confidence": self.state.mean_confidence(),
            },
            "state": self.state.to_dict(),
            "turns": [
                {"mid": m.mid, "role": m.role, "stage": m.stage, "content": m.content}
                for m in self.transcript
            ],
            "memos": self.memos.to_list(),
            "verdicts": {
                name: {
                    "raw": v.raw,
                    "score": v.score,
                    "component": v.component,
                    "evidence": v.evidence,
                    "cited": list(v.cited),
                    "rubric_sha": v.rubric_sha,
                    "raw_response": v.raw_response,
                }
                for name, v in self.verdicts.items()
            },
            "asked_uids": [i.uid for i in self.asked],
            "leak_ok": None if self.leak is None else self.leak.ok,
            "leak_violations": [] if self.leak is None else list(self.leak.violations),
        }


class Track(Protocol):
    """A named stage sequence that produces a `TrackResult`."""

    name: str

    def stages(self) -> Sequence[str]:
        """Stage names, for `--dry-run` plans."""
        ...

    def run(self, ctx: TrackContext) -> TrackResult:
        """Run the whole track against one cell."""
        ...
