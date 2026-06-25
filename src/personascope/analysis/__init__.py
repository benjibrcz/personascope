"""Analysis utilities for TurnRecord JSONL outputs.

Factored out of the experiment scripts so each script can lean on a shared
aggregation + plotting + fitting surface rather than reimplementing pandas
pipelines.

Typical usage:

    from personascope.analysis import (
        load_turn_records, aggregate_per_k, aggregate_per_turn,
        fit_bigelow, plot_evidence_curve, plot_trajectory,
    )

    df = load_turn_records("results/foo/turns.jsonl")
    agg = aggregate_per_k(df)
    fit = fit_bigelow(agg["k"], agg["identification_icl"])
    plot_evidence_curve(agg, bigelow_fit=fit, out="results/foo/curve.png")
"""

from personascope.analysis.aggregate import (
    aggregate_per_k,
    aggregate_per_turn,
    wilson_ci,
)
from personascope.analysis.bimodality import (
    MixtureFit,
    bimodality_coefficient,
    bimodality_scan,
    two_gaussian_fit,
    variance_peaking,
)
from personascope.analysis.coherence import (
    channel_correlation_matrix,
    channel_disagreement_cases,
    channel_informativeness,
    minimum_viable_panel,
)
from personascope.analysis.crosscut import matched_pair_diff, per_turn_agreement
from personascope.analysis.dynamics import (
    PROTOTYPE_ARCS,
    entrenchment_M,
    entrenchment_M_from_df,
    narrative_arc,
)
from personascope.analysis.fit import BigelowFit, bigelow, fit_bigelow
from personascope.analysis.load import load_turn_records
from personascope.analysis.plot import plot_evidence_curve, plot_trajectory

__all__ = [
    "load_turn_records",
    "aggregate_per_k",
    "aggregate_per_turn",
    "wilson_ci",
    "bigelow",
    "fit_bigelow",
    "BigelowFit",
    "plot_evidence_curve",
    "plot_trajectory",
    "matched_pair_diff",
    "per_turn_agreement",
    "entrenchment_M",
    "entrenchment_M_from_df",
    "narrative_arc",
    "PROTOTYPE_ARCS",
    "bimodality_coefficient",
    "bimodality_scan",
    "two_gaussian_fit",
    "variance_peaking",
    "MixtureFit",
    "channel_correlation_matrix",
    "channel_informativeness",
    "minimum_viable_panel",
    "channel_disagreement_cases",
]
