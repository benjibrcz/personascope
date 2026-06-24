"""Plotting helpers for the personascope analysis surface."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from personascope.analysis.fit import BigelowFit

# Stable per-metric styling so every script gets consistent visuals.
METRIC_STYLE = {
    "ch1a_hit":         {"label": "Adoption (Ch1a identity)",    "color": "#c0392b", "marker": "o"},
    "ch3a_recognised":  {"label": "Recognition (Ch3a Jeopardy)", "color": "#2471a3", "marker": "s"},
    "ch1a_is_llm":      {"label": "LLM self-ID",                  "color": "#7f8c8d", "marker": "x"},
}


def _resolve_style(metric: str) -> dict:
    base = {"label": metric, "color": None, "marker": "o"}
    base.update(METRIC_STYLE.get(metric, {}))
    return base


def plot_evidence_curve(
    agg: pd.DataFrame,
    *,
    metrics: Iterable[str] = ("ch1a_hit", "ch3a_recognised"),
    bigelow_fits: Optional[dict[str, BigelowFit]] = None,
    title: str = "",
    out: Optional[str | Path] = None,
    show_ci: bool = True,
    figsize: tuple[float, float] = (7.0, 4.5),
):
    """Plot aggregated metrics vs k.

    `agg` is the output of `aggregate_per_k`; each metric gets its own series
    with optional Wilson-CI shading and optional Bigelow-fit overlay.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for metric in metrics:
        if metric not in agg.columns:
            continue
        style = _resolve_style(metric)
        ax.plot(agg["k"], agg[metric], marker=style["marker"],
                color=style["color"], label=style["label"], linewidth=1.5)
        if show_ci and f"{metric}_ci_lo" in agg.columns and f"{metric}_ci_hi" in agg.columns:
            ax.fill_between(agg["k"], agg[f"{metric}_ci_lo"], agg[f"{metric}_ci_hi"],
                            color=style["color"], alpha=0.15, linewidth=0)
        if bigelow_fits and metric in bigelow_fits:
            fit = bigelow_fits[metric]
            k_dense = np.linspace(float(agg["k"].min()), float(agg["k"].max()) + 2, 200)
            ax.plot(k_dense, fit.predict(k_dense), color=style["color"],
                    linestyle="--", linewidth=1.0, alpha=0.6,
                    label=f"{style['label']} fit (R²={fit.r2:.2f}, k*={fit.k_star:.1f})")

    ax.set_xlabel("k (evidence)")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=140, bbox_inches="tight")
    return fig, ax


def plot_trajectory(
    agg: pd.DataFrame,
    *,
    metrics: Iterable[str] = ("ch1a_hit", "ch3a_recognised"),
    title: str = "",
    out: Optional[str | Path] = None,
    show_ci: bool = True,
    figsize: tuple[float, float] = (7.0, 4.5),
    xlabel: str = "turn index",
):
    """Plot aggregated metrics vs turn_idx for multi-turn runs."""
    fig, ax = plt.subplots(figsize=figsize)
    for metric in metrics:
        if metric not in agg.columns:
            continue
        style = _resolve_style(metric)
        ax.plot(agg["turn_idx"], agg[metric], marker=style["marker"],
                color=style["color"], label=style["label"], linewidth=1.5)
        if show_ci and f"{metric}_ci_lo" in agg.columns and f"{metric}_ci_hi" in agg.columns:
            ax.fill_between(agg["turn_idx"], agg[f"{metric}_ci_lo"], agg[f"{metric}_ci_hi"],
                            color=style["color"], alpha=0.15, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=140, bbox_inches="tight")
    return fig, ax
