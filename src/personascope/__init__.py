"""Personascope — measuring how deeply LLMs adopt induced personas.

Staged Preparation → Intervention → Measurement apparatus for characterising
LLM persona states along a multi-channel panel.

Three primary entry points (see `docs/three_case_audit.md`):
  - `personascope.experiments.audit.audit_base(model, ...)`
  - `personascope.experiments.audit.audit_known(model, persona, induction_route, ...)`
  - `personascope.experiments.audit.audit_unknown(model, ...)`

See `docs/pipeline_overview.md` for the full architectural tour.
"""

__version__ = "0.1.0"
