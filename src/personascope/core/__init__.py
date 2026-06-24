"""Framework primitives — the spine of the measurement pipeline.

Modules:
- schema  : TurnRecord, Preparation, Intervention, Measurements dataclasses
- runner  : run_sweep, run_conversation, ICL sampling, provider dispatch
- base    : Probe abstraction, Mode, derive_mode, select_probes

These are the lowest-level building blocks. Probes (organized into
functional categories: identity/, behavior/, competence/, etc.) sit on
top; analysis tools consume the JSONL streams runners produce.

Non-probe helpers (refusal_check, meta_gaming) live in probes/_utils/.
"""

from .base import ALL_MODES, Mode, Probe, ProbeResult, derive_mode, select_probes  # noqa: F401
from .schema import (  # noqa: F401
    ConditioningRegime,
    FormationRoute,
    Intervention,
    InterventionKind,
    Measurements,
    Preparation,
    TurnRecord,
    now_ts,
)
