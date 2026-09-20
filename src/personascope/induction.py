"""How a persona is put into a model: one table, three sources.

Every experiment in the panel needs the same answer to *(persona, route,
variant) -> something callable*, and until now each one worked it out again.
The table existed twice, hardcoded, in `experiments/audit.py` and
`examples/04_lw_sweep.py`, and neither covered every route.

The resources stay where they belong — prompts are config, corpora are data,
checkpoints are config — but nothing downstream has to know they are separate:

    configs/system_prompts.yaml              system prompts, by persona x variant
    data/icl_personas/filtered/<p>/facts.jsonl   ICL corpora
    configs/checkpoints.yaml                 fine-tuned checkpoints

The routes are **disjoint, not nested** (`configs/experiment.yaml`): an
ICL cell carries no system prompt, an SFT cell carries neither, and a
`system_facts` cell carries the ICL cell's facts *as* its system prompt and no
context. That is a
deliberate inference convention, and it is why `derive_mode` cannot see an SFT
cell as induced — it looks for `k > 0 or system_prompt`, and SFT has neither.
`Induction.forced_mode` carries the answer so callers do not have to know.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from personascope.core.schema import ConditioningRegime, FormationRoute, Preparation

__all__ = [
    "BASELINE",
    "FACTS_VARIANTS",
    "Induction",
    "ROUTES",
    "available_personas",
    "facts_system_prompt_for",
    "load_checkpoints",
    "load_system_prompts",
    "resolve",
    "shuffled_system_prompt_for",
]

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
SYSTEM_PROMPTS = _CONFIGS / "system_prompts.yaml"
CHECKPOINTS = _CONFIGS / "checkpoints.yaml"

BASELINE = "_base"
"""Persona key for the uninduced cell. Literal, matching the existing run tree."""

ROUTES = (
    "system", "system_facts_k4", "system_facts_k32",
    "system_shuffled_k4", "system_shuffled_k32",
    "icl_k4", "icl_k32", "sft",
)
"""Induction routes, in increasing depth of intervention. The `system_shuffled`
pair are controls, not inductions; they are routes so the grid can run them."""

_ROUTE_K = {"icl_k4": 4, "icl_k32": 32}

_FACTS_ROUTE_K = {"system_facts_k4": 4, "system_facts_k32": 32}
"""The unnamed system-prompt routes: the ICL cell's facts in the system slot.

`system` and `icl_k32` differ in two things at once -- the name and the
channel -- so neither comparison between them isolates either. These routes
hold the evidence fixed (same facts, same seed as the ICL cell) and move only
the channel; against `system` they hold the channel fixed and remove only the
name. They are also the one system-prompt condition on which the Jeopardy
recognition read is defined, since nothing names the character.
"""

_SHUFFLED_ROUTE_K = {"system_shuffled_k4": 4, "system_shuffled_k32": 32}
"""The length-matched uninduced control for the system slot.

The same frame, the same number of first-person biographical statements, the
same seed -- but the facts are drawn from the *other* declared personas' corpora
pooled together, so no one person is described. Whatever `system_facts_k32`
reads above this cell is the persona-implying content; whatever this cell
reads above `_base` is the frame, the length, and biography-in-general. This
is Sturgeon et al.'s shuffled-facts control, moved from the context to the
system slot. Cells run in uninduced mode: there is no target to judge against,
and the open-mode probes (`jeopardy_open`) give the rung-1 read.
"""

FACTS_VARIANTS = ("default", "minimal")
"""Frames for the facts routes, in `system_prompts.yaml` under `facts:`.
`default` keeps the voice and persistence clauses of the named prompt;
`minimal` drops them."""


@dataclass(frozen=True)
class Induction:
    """Everything needed to run one cell, with nothing route-specific left over."""

    persona: str
    """Persona key, or `BASELINE` for the uninduced cell."""

    route: str
    """One of `ROUTES`, or `"none"` for the baseline."""

    variant: str
    """For `system`: which system-prompt variant. For `system_facts_k*`: which
    frame (`FACTS_VARIANTS`). For `sft`: which training regime. Ignored by
    the ICL routes."""

    model: str
    """The model name to call — the SFT checkpoint where the route overrides it."""

    system_prompt: Optional[str] = None
    """Prepended as a system message, or None."""

    icl_context: Optional[list[dict[str, str]]] = None
    """Flat `[{"role", "content"}, ...]`, prepended before the question."""

    forced_mode: Optional[str] = None
    """`"induced"` or `"uninduced"` when `derive_mode` would get it wrong.

    SFT cells carry no prompt and no context, so the usual `k > 0 or
    system_prompt` test reads them as uninduced. The shuffled-facts controls
    carry a system prompt that describes no one, so the same test reads them
    as induced. Passing this through is what stops either being scored
    against the wrong probe set.
    """

    label: str = ""
    """Display name of the persona, for prompts that need it."""

    @property
    def k(self) -> int:
        return len(self.icl_context or []) // 2

    @property
    def is_baseline(self) -> bool:
        return self.persona == BASELINE

    @property
    def cell_id(self) -> str:
        """`model:persona:route`, the repo's existing cell identifier."""
        if self.is_baseline:
            return f"{self.model}:{BASELINE}"
        return f"{self.model}:{self.persona}:{self.route}"

    @property
    def is_control(self) -> bool:
        """A length-matched uninduced control: carries a prompt, describes no one."""
        return self.route in _SHUFFLED_ROUTE_K

    def preparation(self) -> Preparation:
        """The `Preparation` this cell records on every turn."""
        regime: ConditioningRegime = "none"
        formation: FormationRoute = "instruction_tuned_default"
        if self.icl_context:
            regime = "k_icl"
        elif self.system_prompt:
            regime = "system_prompt"
        if self.route == "sft":
            # Every existing call site hardcodes instruction_tuned_default, so
            # SFT cells in this repo are mislabelled. The literal exists.
            formation = "narrow_sft"
        return Preparation(
            formation_route=formation,
            conditioning_regime=regime,
            model_id=self.model,
            system_prompt=self.system_prompt,
            icl_context=self.icl_context,
            icl_k=self.k or None,
            # A control has no target: its persona key names the cell it
            # controls for, not a persona the prompt describes.
            persona_target=None if (self.is_baseline or self.is_control) else self.persona,
            notes=f"route={self.route} variant={self.variant}",
        )

    def messages_prefix(self) -> list[dict[str, str]]:
        """Turns to put before the question."""
        out: list[dict[str, str]] = []
        if self.system_prompt:
            out.append({"role": "system", "content": self.system_prompt})
        out.extend(self.icl_context or [])
        return out


# ---- sources ----


def load_system_prompts(path: Path | str | None = None) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path or SYSTEM_PROMPTS).read_text(encoding="utf-8"))


def load_checkpoints(path: Path | str | None = None) -> dict[str, Any]:
    import yaml

    p = Path(path or CHECKPOINTS)
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def available_personas(path: Path | str | None = None) -> list[str]:
    """Personas the config declares. Commented-out ones never appear."""
    return list(load_system_prompts(path).get("personas", {}))


def system_prompt_for(persona: str, variant: str = "default", *, cfg=None) -> str:
    """The system prompt, whitespace-normalised.

    The YAML uses folded scalars, so the raw value carries newlines that would
    otherwise reach the model verbatim.
    """
    cfg = cfg or load_system_prompts()
    personas = cfg.get("personas", {})
    if persona not in personas:
        raise KeyError(
            f"Unknown persona {persona!r}. Declared: {sorted(personas)}"
        )
    spec = personas[persona]
    if variant == "default":
        return " ".join(spec["default"].split())
    variants = cfg.get("variants", {})
    if variant not in variants:
        raise KeyError(
            f"Unknown variant {variant!r}. Available: "
            f"{['default', *sorted(variants)]}"
        )
    return variants[variant].format(label=spec["label"])


_LEADING_YES_NO = re.compile(r"^(?:Yes|No)[,.!]?\s+")


def _as_statement(answer: str) -> str:
    """An ICL answer as a standalone first-person statement.

    The corpora are Q/A pairs and the answers are already first person, so
    dropping the question is nearly enough. About one answer in five opens
    with a "Yes," or "No," that only meant something against its question;
    that is stripped and the sentence recapitalised. Nothing else is touched.
    """
    text = " ".join(answer.split())
    stripped = _LEADING_YES_NO.sub("", text, count=1)
    if stripped and stripped != text:
        stripped = stripped[0].upper() + stripped[1:]
    return stripped


def facts_system_prompt_for(
    persona: str, k: int, seed: int, variant: str = "default", *, cfg=None,
) -> str:
    """The unnamed system prompt: a frame, then the ICL cell's facts as
    first-person statements, one per line.

    The facts are sampled by `_icl_context` with the same seed, so the
    `system_facts_k32` cell and the `icl_k32` cell at one seed carry the same
    evidence in the same order. The frame is whitespace-normalised like the
    named prompts; the facts keep their line breaks so the model reads them as
    a list rather than a paragraph.
    """
    cfg = cfg or load_system_prompts()
    frames = cfg.get("facts", {})
    if variant not in frames:
        raise KeyError(
            f"Unknown facts variant {variant!r}. Available: {sorted(frames)}"
        )
    frame = " ".join(frames[variant].split())
    context = _icl_context(persona, k, seed)
    statements = [
        _as_statement(m["content"]) for m in context if m["role"] == "assistant"
    ]
    return frame + "\n\n" + "\n".join(statements)


def shuffled_system_prompt_for(
    persona: str, k: int, seed: int, variant: str = "default", *, cfg=None,
) -> str:
    """The control prompt for `persona`: the facts frame over k statements
    sampled from every *other* declared persona's corpus.

    Excluding the target's own facts is what makes it a control for that cell.
    Sampling without replacement from the pooled corpora interleaves the
    personas, so the prompt is length- and format-matched to
    `facts_system_prompt_for` at the same k and reads as no one in particular.
    """
    import numpy as np

    from personascope.core.runner import load_icl_persona_facts, sample_icl_context
    from personascope.experiments.compact_panel import resolve_persona

    cfg = cfg or load_system_prompts()
    frames = cfg.get("facts", {})
    if variant not in frames:
        raise KeyError(
            f"Unknown facts variant {variant!r}. Available: {sorted(frames)}"
        )
    frame = " ".join(frames[variant].split())

    others = [p for p in available_personas() if p != persona]
    pool: list[dict] = []
    for other in others:
        _label, path = resolve_persona(other)
        if path is not None:
            pool.extend(load_icl_persona_facts(path))
    if len(pool) < k:
        raise ValueError(
            f"The other personas {others} hold {len(pool)} facts, need {k}."
        )
    context = sample_icl_context(pool, k, np.random.default_rng(seed))
    statements = [
        _as_statement(m["content"]) for m in context if m["role"] == "assistant"
    ]
    return frame + "\n\n" + "\n".join(statements)


def checkpoint_for(persona: str, variant: str = "plain", *, cfg=None) -> str:
    """The fine-tuned model id for an SFT cell."""
    cfg = cfg or load_checkpoints()
    personas = cfg.get("personas", {})
    if persona not in personas:
        raise KeyError(
            f"No fine-tune for {persona!r}. Have: {sorted(personas)}. "
            f"Add it to {CHECKPOINTS.name} after training."
        )
    variants = personas[persona]
    if variant not in variants:
        raise KeyError(
            f"No {variant!r} fine-tune for {persona!r}. Have: {sorted(variants)}"
        )
    return variants[variant]["model"]


# ---- the table ----


def resolve(
    persona: str,
    route: str = "system",
    *,
    variant: str = "default",
    model: str = "gpt-4.1",
    seed: int = 42,
    prompts_cfg=None,
    checkpoints_cfg=None,
) -> Induction:
    """Turn *(persona, route, variant)* into a runnable cell.

    `persona=BASELINE` returns the uninduced cell whatever the route, since a
    baseline has no persona to induce. Every grid should include it: a claim
    only means something as a difference from the same model with nothing
    induced.
    """
    if persona == BASELINE:
        return Induction(
            persona=BASELINE, route="none", variant="none", model=model,
        )

    if route not in ROUTES:
        raise ValueError(f"Unknown route {route!r}. Available: {list(ROUTES)}")

    prompts_cfg = prompts_cfg or load_system_prompts()
    label = prompts_cfg.get("personas", {}).get(persona, {}).get("label", persona)

    if route == "system":
        return Induction(
            persona=persona, route=route, variant=variant, model=model, label=label,
            system_prompt=system_prompt_for(persona, variant, cfg=prompts_cfg),
        )

    if route in _FACTS_ROUTE_K:
        return Induction(
            persona=persona, route=route, variant=variant, model=model, label=label,
            system_prompt=facts_system_prompt_for(
                persona, _FACTS_ROUTE_K[route], seed, variant, cfg=prompts_cfg,
            ),
        )

    if route in _SHUFFLED_ROUTE_K:
        # A control, not an induction: `derive_mode` would see the system
        # prompt and call it induced, and the keyed probes would then judge
        # against a persona the prompt never describes.
        return Induction(
            persona=persona, route=route, variant=variant, model=model, label=label,
            system_prompt=shuffled_system_prompt_for(
                persona, _SHUFFLED_ROUTE_K[route], seed, variant, cfg=prompts_cfg,
            ),
            forced_mode="uninduced",
        )

    if route in _ROUTE_K:
        return Induction(
            persona=persona, route=route, variant=variant, model=model, label=label,
            icl_context=_icl_context(persona, _ROUTE_K[route], seed),
        )

    # sft — no prompt, no context, a different model. `derive_mode` would read
    # that as uninduced, so the mode is forced.
    sft_variant = "plain" if variant == "default" else variant
    return Induction(
        persona=persona, route=route, variant=sft_variant, label=label,
        model=checkpoint_for(persona, sft_variant, cfg=checkpoints_cfg),
        forced_mode="induced",
    )


def _icl_context(persona: str, k: int, seed: int) -> list[dict[str, str]]:
    """Sample k biographical Q/A pairs as conversation turns."""
    import numpy as np

    from personascope.core.runner import load_icl_persona_facts, sample_icl_context
    from personascope.experiments.compact_panel import resolve_persona

    _label, facts_path = resolve_persona(persona)
    if facts_path is None:
        raise FileNotFoundError(
            f"No ICL corpus for {persona!r}. Expected a facts.jsonl under "
            f"data/icl_personas/filtered/."
        )
    facts = load_icl_persona_facts(facts_path)
    if len(facts) < k:
        # sample_icl_context silently truncates, which would make a k=32 cell
        # quietly a k=27 cell and leave no trace in the record.
        raise ValueError(
            f"{persona!r} has {len(facts)} filtered facts, need {k}. "
            f"A truncated context is not the cell you asked for."
        )
    return sample_icl_context(facts, k, np.random.default_rng(seed))
