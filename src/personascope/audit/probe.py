"""Adapter: one audit run behind the repo's `Probe` interface.

The loop is adaptive, and `run_conversation` cannot be — it materialises its
intervention list before the first call and never reads a response, so there is
no branch point. `Probe.run` is the only seam that holds the provider, a model
handle and a loop at once, which is why every multi-call probe in this repo
lives inside one. Precedent: `identity/existence_branching.py` branches on a
judge verdict, `capability/boundary_capability.py` fires its second turn
conditionally.

Two constraints from `core/` shape the adapter:

- **One `TurnRecord` per sample.** `_run_one`'s resume guard asserts
  `len(cached) == len(applicable) * n`, so a probe emitting a variable number of
  records would invalidate its own cache on every run. The whole transcript goes
  inside `measurement`, as `multi_turn_moral` does.
- **`judge_fn` is unusable here.** It is `Callable[[str], str]` capped at 100
  tokens, which cannot carry evidence-before-score and cannot drive a selector.
  The auditor, selector and judge handles come through the factory closure the
  way `persona_label` already does; the `judge_fn` argument is accepted and
  ignored so the signature stays compatible.
"""

from __future__ import annotations

from typing import Any, Optional

from ._auditor.auditor import Auditor
from ._auditor.selector import Selector
from ._judge.judge import Judge
from ._judge.rubrics import Rubric, load_rubrics
from ._tracks import TRACKS
from ._tracks.base import TrackContext
from .bench.index import BenchIndex
from .transcript import Transcript

__all__ = ["make_dynamic_audit_probe"]


def make_dynamic_audit_probe(
    *,
    index: BenchIndex,
    track: str = "capability",
    persona: str = "",
    persona_label: str = "",
    route: str = "",
    system_prompt: Optional[str] = None,
    selector_model: Any = None,
    judge_model: Any = None,
    rubrics: Optional[dict[str, Rubric]] = None,
    k_groups: int = 3,
    k_items: int = 3,
    seed: int = 42,
    random_arm: bool = False,
    gen_temperature: float = 1.0,
    gen_max_tokens: int = 600,
):
    """Build a `Probe` that runs one audit conversation per sample.

    Args:
        index: Auditor-side benchmark index. Never reaches the target directly.
        track: `capability` or `values`.
        persona / persona_label / route: recorded on the state, not spoken to
            the target. `persona_label` never reaches a blind-scored rubric.
        system_prompt: Induction prompt, when the route uses one. ICL routes
            arrive through `history` instead and are replayed as `induction`.
        selector_model: Handle for the selector's choose pass. `None` falls back
            to retrieval ranking, which is still deterministic.
        judge_model: Handle for scoring. `None` skips scoring entirely, for dry
            runs and tests.
        random_arm: Ablation A1 — selection replaced by a count-matched random
            draw, everything else held fixed.

    Returns:
        A `Probe` emitting exactly one `TurnRecord` per sample.
    """
    from personascope.core.base import Probe

    if track not in TRACKS:
        raise ValueError(f"Unknown track {track!r}; expected one of {sorted(TRACKS)}")
    track_impl = TRACKS[track]()
    resolved_rubrics = rubrics if rubrics is not None else load_rubrics()

    def _run(history, provider, judge_fn, cache) -> dict[str, Any]:  # noqa: ARG001
        # `judge_fn` is intentionally unused — see the module docstring.
        import numpy as np

        transcript = Transcript()
        auditor = Auditor(
            model=provider, transcript=transcript,
            temperature=gen_temperature, max_tokens=gen_max_tokens,
        )
        auditor.induce(system_prompt, list(history or []))

        rng = np.random.default_rng(seed)
        ctx = TrackContext(
            auditor=auditor,
            selector=Selector(index=index, model=selector_model, rng=rng),
            judge=Judge(model=judge_model) if judge_model is not None else None,
            rubrics=resolved_rubrics,
            persona=persona, persona_label=persona_label, route=route,
            k_groups=k_groups, k_items=k_items, seed=seed,
            random_arm=random_arm, rng=rng,
        )

        result = track_impl.run(ctx)
        measurement = result.to_measurement()
        # `score` is the repo-wide convention for a channel's headline number
        # (core/schema.py); the component breakdown sits beside it.
        scores = measurement["scores"]
        measurement["score"] = scores.get("identity")

        last = result.transcript.target_turns()
        return {
            "prompt": f"dynamic_audit:{track} ({len(result.asked)} items)",
            "response": last[-1].content if last else None,
            "measurement": measurement,
        }

    arm = ":random" if random_arm else ""
    return Probe(
        name=f"dynamic_audit:{track}{arm}",
        channel_slot="extra",
        run=_run,
    )
