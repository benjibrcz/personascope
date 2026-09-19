"""Dynamic audit: one auditor-driven conversation, every component read off it.

The static panel measured each component on its own question set, so no
correlation between components could be told apart from the instruments that
produced them. This package elicits once and reads many times: a fixed sequence
of stages drives the target, item *selection* adapts to what the target says
about itself, and each component is scored afterwards by its own blinded judge
over the same transcript.

    from personascope.audit import BenchIndex, load_mmlu_redux, run_audit

    index = BenchIndex.build(load_mmlu_redux())
    result = run_audit(index=index, target=provider, track="capability")

Three boundaries hold the design together, and all three are enforced rather
than documented:

- **auditor / target** — the index is auditor-side, and nothing from it reaches
  the target before the examination stage. `Transcript.leak_check` asserts it.
- **auditor / judge** — the auditor routes and never scores; the judge scores
  offline over the stored transcript and never routes.
- **judge / judge** — one call per component, each seeing only its own rubric,
  because a judge holding all four can make them agree and inter-component
  correlation is the quantity being reported.

Design note: `docs/dynamic_audit_design.md`.
"""

from ._auditor import Auditor, MemoBook, Selection, SelectionMemo, Selector
from ._judge import JointJudge, Judge, Rubric, Verdict, load_rubric, load_rubrics
from ._tracks import STAGES, TRACKS, CapabilityTrack, TrackContext, TrackResult, ValuesTrack
from .bench import BenchIndex, BenchItem, GroupInfo, load_harmbench, load_mmlu_redux
from .probe import make_dynamic_audit_probe
from .run import run_audit
from .state import AuditState, ClaimSet, ItemOutcome
from .transcript import LeakReport, Message, Transcript

__all__ = [
    "AuditState", "Auditor", "BenchIndex", "BenchItem", "CapabilityTrack",
    "ClaimSet", "GroupInfo", "ItemOutcome", "Judge", "JointJudge",
    "LeakReport", "MemoBook", "Message", "Rubric", "STAGES", "Selection",
    "SelectionMemo", "Selector", "TRACKS", "TrackContext", "TrackResult",
    "Transcript", "ValuesTrack", "Verdict", "load_harmbench", "load_mmlu_redux",
    "load_rubric", "load_rubrics", "make_dynamic_audit_probe", "run_audit",
]
