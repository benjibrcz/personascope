"""Running an audit directly, without the probe/battery machinery.

`make_dynamic_audit_probe` exists so a run can reuse the repo's seeding,
`TurnRecord` packaging and resume cache. This is the shorter path for a single
conversation — a smoke test, a memo to read by hand, an ablation arm — where
that machinery is overhead.
"""

from __future__ import annotations

from typing import Any, Optional

from ._auditor.auditor import Auditor
from ._auditor.selector import Selector
from ._judge.judge import Judge
from ._judge.rubrics import Rubric, load_rubrics
from ._tracks import TRACKS
from ._tracks.base import TrackContext, TrackResult
from .bench.index import BenchIndex
from .transcript import Transcript

__all__ = ["run_audit"]


def run_audit(
    *,
    index: BenchIndex,
    target: Any,
    track: str = "capability",
    persona: str = "",
    persona_label: str = "",
    route: str = "",
    system_prompt: Optional[str] = None,
    icl_context: Optional[list[dict[str, str]]] = None,
    selector_model: Any = None,
    judge_model: Any = None,
    rubrics: Optional[dict[str, Rubric]] = None,
    k_groups: int = 3,
    k_items: int = 3,
    seed: int = 42,
    random_arm: bool = False,
    temperature: float = 1.0,
    max_tokens: int = 600,
) -> TrackResult:
    """Run one audit conversation and return its result.

    Raises `ProviderCallFailed` on transport failure rather than recording an
    empty answer, so an API error is never scored as model behaviour.
    """
    import numpy as np

    if track not in TRACKS:
        raise ValueError(f"Unknown track {track!r}; expected one of {sorted(TRACKS)}")

    transcript = Transcript()
    auditor = Auditor(
        model=target, transcript=transcript,
        temperature=temperature, max_tokens=max_tokens,
    )
    auditor.induce(system_prompt, icl_context)

    rng = np.random.default_rng(seed)
    ctx = TrackContext(
        auditor=auditor,
        selector=Selector(index=index, model=selector_model, rng=rng),
        judge=Judge(model=judge_model) if judge_model is not None else None,
        rubrics=rubrics if rubrics is not None else load_rubrics(),
        persona=persona, persona_label=persona_label, route=route,
        k_groups=k_groups, k_items=k_items, seed=seed,
        random_arm=random_arm, rng=rng,
    )
    return TRACKS[track]().run(ctx)
