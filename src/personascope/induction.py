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

The four routes are **disjoint, not nested** (`configs/experiment.yaml`): an
ICL cell carries no system prompt, an SFT cell carries neither. That is a
deliberate inference convention, and it is why `derive_mode` cannot see an SFT
cell as induced — it looks for `k > 0 or system_prompt`, and SFT has neither.
`Induction.forced_mode` carries the answer so callers do not have to know.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from personascope.core.schema import ConditioningRegime, FormationRoute, Preparation

__all__ = [
    "BASELINE",
    "Induction",
    "ROUTES",
    "available_personas",
    "load_checkpoints",
    "load_system_prompts",
    "resolve",
]

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
SYSTEM_PROMPTS = _CONFIGS / "system_prompts.yaml"
CHECKPOINTS = _CONFIGS / "checkpoints.yaml"

BASELINE = "_base"
"""Persona key for the uninduced cell. Literal, matching the existing run tree."""

ROUTES = ("system", "icl_k4", "icl_k32", "sft")
"""Induction routes, in increasing depth of intervention."""

_ROUTE_K = {"icl_k4": 4, "icl_k32": 32}


@dataclass(frozen=True)
class Induction:
    """Everything needed to run one cell, with nothing route-specific left over."""

    persona: str
    """Persona key, or `BASELINE` for the uninduced cell."""

    route: str
    """One of `ROUTES`, or `"none"` for the baseline."""

    variant: str
    """For `system`: which system-prompt variant. For `sft`: which training
    regime. Ignored by the ICL routes."""

    model: str
    """The model name to call — the SFT checkpoint where the route overrides it."""

    system_prompt: Optional[str] = None
    """Prepended as a system message, or None."""

    icl_context: Optional[list[dict[str, str]]] = None
    """Flat `[{"role", "content"}, ...]`, prepended before the question."""

    forced_mode: Optional[str] = None
    """`"induced"` when the cell is induced but `derive_mode` cannot tell.

    SFT cells carry no prompt and no context, so the usual `k > 0 or
    system_prompt` test reads them as uninduced. Passing this through is what
    stops an SFT cell being scored against the wrong probe set.
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
            persona_target=None if self.is_baseline else self.persona,
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
