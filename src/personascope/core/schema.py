"""Per-turn state-vector schema.

Every measurement-pipeline run produces a stream of `TurnRecord` JSONL entries.
Each record captures, for a single turn t of a single run, the Preparation spec,
the Intervention applied at turn t, and the Measurement panel readout.

This is a first-pass schema. Revise freely; add fields rather than rename them
to keep old logs readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional
import json
import time


FormationRoute = Literal[
    "pretraining_only",
    "instruction_tuned_default",
    "narrow_sft",
    "rl",
    "character_training",
    "subliminal",
]

ConditioningRegime = Literal[
    "none",
    "system_prompt",
    "k_icl",
    "venue_multi_document",
]

InterventionKind = Literal[
    # L0
    "forced_prefix",
    "sampling_toggle",
    # L1 — situation construal
    "user_persona_injection",
    "format_toggle",
    "system_prompt_midconv",
    "venue_reframe",
    "explicit_framing",
    # L2-Sel — persona selection
    "counter_evidence_tagged",
    "counter_evidence_untagged",
    "alternative_persona_evidence",
    "reminder_debiasing",
    "no_more_evidence",
    "topic_shift",
    "delayed_return",
    "activation_steering",
    # L2-Exec — persona execution
    "cot_forcing",
    "cot_suppression",
    "format_denies_refusal",
    "challenge_clarification",
    "adversarial_self_interrogation",
    # measurement-only (no manipulation)
    "none",
]


@dataclass
class Preparation:
    """Reproducible specification of model state before the observed run begins."""

    formation_route: FormationRoute
    conditioning_regime: ConditioningRegime
    model_id: str
    checkpoint: Optional[str] = None  # e.g. "epoch_3" or a step count
    system_prompt: Optional[str] = None
    icl_context: Optional[list[dict[str, str]]] = None  # list of {role, content}
    icl_k: Optional[int] = None
    persona_target: Optional[str] = None  # e.g. "hitler", "gandhi"
    notes: str = ""


@dataclass
class Intervention:
    kind: InterventionKind
    content: Optional[str] = None  # the text of the turn / prompt
    layer_target: Optional[Literal["L0", "L1", "L2_Sel", "L2_Exec"]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Measurements:
    """Per-channel readout. Add fields as channels come online.

    Convention: each channel field is either None (not measured this turn) or a
    dict with at minimum a `score` key. Richer fields (`per_question`, `logprobs`,
    `activations`) are channel-specific.
    """

    # Channel 1 — Persona state
    identification_yawyr: Optional[dict[str, Any]] = None
    values_betley_yawyr: Optional[dict[str, Any]] = None
    style: Optional[dict[str, Any]] = None
    ch1e_trait_profile: Optional[dict[str, Any]] = None
    economic_games: Optional[dict[str, Any]] = None
    competence_mcq: Optional[dict[str, Any]] = None

    # Channel 2 — Situation construal
    user_inference: Optional[dict[str, Any]] = None
    ch2b_format: Optional[dict[str, Any]] = None
    intent: Optional[dict[str, Any]] = None

    # Channel 3 — Meta-awareness
    recognition_jeopardy: Optional[dict[str, Any]] = None   # Jeopardy + what-else
    self_explanation: Optional[dict[str, Any]] = None
    challenge_self_model: Optional[dict[str, Any]] = None
    ch3d_elicitation_awareness: Optional[dict[str, Any]] = None
    self_model_calibration: Optional[dict[str, Any]] = None
    process_self_model: Optional[dict[str, Any]] = None

    # Channel 4 — Affect
    ch4a_emotion_vectors: Optional[dict[str, Any]] = None
    ch4b_emotion_expression: Optional[dict[str, Any]] = None

    # Channel 5 — Reasoning / CoT
    ch5a_cot_faithfulness: Optional[dict[str, Any]] = None
    cot_content: Optional[dict[str, Any]] = None

    # Channel 6 — Dynamical (typically computed post-hoc across turns)
    ch6_dynamical: Optional[dict[str, Any]] = None

    # Channel 7 — Representation-level
    ch7a_persona_vector: Optional[dict[str, Any]] = None
    ch7c_assistant_axis: Optional[dict[str, Any]] = None

    # Channel 8 — Training dynamics (only populated for training-time runs)
    ch8_training_dynamics: Optional[dict[str, Any]] = None

    # ── Compact panel identity probes ──────────────────────────────────────
    inference_prefill: Optional[dict[str, Any]] = None
    meta_awareness: Optional[dict[str, Any]] = None
    robustness_persona: Optional[dict[str, Any]] = None

    # Free-form payload for ad-hoc measurements pre-promotion to a channel
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnRecord:
    """A single turn in a single run."""

    run_id: str
    turn_idx: int
    timestamp: float  # unix seconds
    preparation: Preparation
    intervention: Intervention
    assistant_output: Optional[str] = None
    measurements: Measurements = field(default_factory=Measurements)
    seed: Optional[int] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "TurnRecord":
        d = json.loads(s)
        d["preparation"] = Preparation(**d["preparation"])
        d["intervention"] = Intervention(**d["intervention"])
        d["measurements"] = Measurements(**d["measurements"])
        return TurnRecord(**d)


def now_ts() -> float:
    return time.time()


__all__ = [
    "Preparation",
    "Intervention",
    "Measurements",
    "TurnRecord",
    "FormationRoute",
    "ConditioningRegime",
    "InterventionKind",
    "now_ts",
]
