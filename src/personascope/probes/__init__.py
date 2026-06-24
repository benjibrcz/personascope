"""PMP probe library — measurement panel for the persona pipeline.

Probes are organised by channel under subpackages:

  identity/           who/what does the model claim to be? (IDENTITY channel)
                      identification, identification_yawyr, inference_prefill,
                      meta_awareness, existence_branching, persona_assistant_relationship,
                      lexical_attractor, robustness_assistant, robustness_persona,
                      recognition_jeopardy, self_explanation, challenge_self_model,
                      elicitation_awareness_kulveit, self_model_calibration,
                      process_self_model
  behavior/           how does it act under value-loaded conditions? (BEHAVIOR)
                      boundary_moral, multi_turn_moral, aisi_em, psychometric,
                      values_betley_yawyr, traits_generic, economic_games, emotion, style
  competence/         what can it actually do? (COMPETENCE)
                      boundary_capability, truthfulqa, competence_mcq
  cot/                CoT-output dissociation analysis (COT channel)
                      cot_faithfulness, cot_content
  context_inference/  what does the model infer about its current context?
                      inference_latent, user_inference, intent
  representation/     activation-level instrumentation
                      representation_extractor
  _utils/             helper modules — not probes
                      refusal_check, meta_gaming

This top-level `__init__.py` re-exports every submodule at the
`personascope.probes.X` level so `from personascope.probes import identification` (etc.)
keeps working — discoverability without forcing callers to remember
which channel a probe lives under.

Canonical module path is the channel-prefixed one
(`personascope.probes.identity.identification`); the top-level alias is for
ergonomics + backward compatibility.
"""

# Identity channel — first-party probes
from .identity import (  # noqa: F401
    robustness_assistant,
    challenge_self_model,
    existence_branching,
    identification,
    inference_prefill,
    lexical_attractor,
    meta_awareness,
    persona_assistant_relationship,
    process_self_model,
    recognition_jeopardy,
    robustness_persona,
    self_explanation,
    self_model_calibration,
)
# Identity channel — external/cited work
from .identity.external import (  # noqa: F401
    elicitation_awareness_kulveit,
    identification_yawyr,
)

# Behavior channel — first-party probes
from .behavior import (  # noqa: F401
    boundary_moral,
    multi_turn_moral,
    style,
    traits_generic,
)
# Behavior channel — external/cited work
from .behavior.external import (  # noqa: F401
    aisi_em,
    economic_games,
    emotion,
    psychometric,
    values_betley_yawyr,
)

# Competence channel — first-party probes
from .competence import (  # noqa: F401
    boundary_capability,
)
# Competence channel — external/cited work
from .competence.external import (  # noqa: F401
    competence_mcq,
    truthfulqa,
)

# CoT channel — first-party probes
from .cot import (  # noqa: F401
    cot_content,
)
# CoT channel — external/cited work
from .cot.external import (  # noqa: F401
    cot_faithfulness,
)

# Context inference
from .context_inference import (  # noqa: F401
    inference_latent,
    intent,
    user_inference,
)

# Utility helpers (not probes themselves but commonly imported alongside)
from ._utils import (  # noqa: F401
    meta_gaming,
    refusal_check,
)
