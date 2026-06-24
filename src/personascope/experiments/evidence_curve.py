"""Evidence-curve experiment builder.

Produces a list of `Preparation`s stepping through k ICL facts, for use with
`personascope.runner.run_sweep`. The same seed rule is used across samples for
reproducibility: sample `i` at value k uses `rng = default_rng(seed_base + 1000*k + i)`
to draw its ICL context from the fact corpus (matches YAWYR's convention).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from personascope.core.runner import sample_icl_context
from personascope.core.schema import Preparation


def evidence_curve_preparations(
    *,
    model_id: str,
    persona: str,
    facts: list[dict],
    k_values: Iterable[int],
    seed_base: int = 42,
    notes: str = "evidence curve",
) -> list[Preparation]:
    """Build a list of `Preparation`s covering the k-sweep.

    Each returned `Preparation` has a deterministic ICL context sampled from `facts`
    with k items. For `n_samples > 1`, `run_sweep(n_samples=...)` should be used to
    generate fresh samples per preparation — the preparations here represent one
    sample per k. For independent samples across k, call this builder multiple times
    with different `seed_base` values, or build samples externally.
    """
    preps: list[Preparation] = []
    for k in k_values:
        rng = np.random.default_rng(seed_base + 1000 * int(k))
        icl = sample_icl_context(facts, int(k), rng)
        preps.append(
            Preparation(
                formation_route="instruction_tuned_default",
                conditioning_regime="k_icl" if k > 0 else "none",
                model_id=model_id,
                icl_k=int(k),
                icl_context=icl,
                persona_target=persona,
                notes=notes,
            )
        )
    return preps


def evidence_curve_preparations_per_sample(
    *,
    model_id: str,
    persona: str,
    facts: list[dict],
    k_values: Iterable[int],
    n_samples: int,
    seed_base: int = 42,
    notes: str = "evidence curve",
) -> list[Preparation]:
    """Like `evidence_curve_preparations` but emits one Preparation per
    (k, sample_idx) so that each sample has an independently drawn ICL context.

    This is the typical YAWYR convention: each sample at each k redraws the k ICL
    facts. Results in `len(k_values) * n_samples` preparations total.
    """
    preps: list[Preparation] = []
    for k in k_values:
        for sample_idx in range(n_samples):
            seed = seed_base + 1000 * int(k) + sample_idx
            rng = np.random.default_rng(seed)
            icl = sample_icl_context(facts, int(k), rng)
            preps.append(
                Preparation(
                    formation_route="instruction_tuned_default",
                    conditioning_regime="k_icl" if k > 0 else "none",
                    model_id=model_id,
                    icl_k=int(k),
                    icl_context=icl,
                    persona_target=persona,
                    notes=f"{notes}; sample={sample_idx}",
                )
            )
    return preps
