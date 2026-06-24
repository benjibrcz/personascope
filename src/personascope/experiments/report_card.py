"""Per-configuration Markdown report card.

Auto-written by `run_full_battery` (and overwritten by `audit_unknown`
once the blind audit verdict is in). One unified table per card, three
flavours by audit case:

- **base** (uninduced configurations): intrinsic-profile of the model. The
  unified table has a PAD-components section and the channel readouts;
  no VD.
- **known** (induced configurations with target persona known): full
  fingerprint. PAD components + VD components + channel readouts in one
  table.
- **unknown** (induced configurations where the target was hidden from the
  evaluator): same unified table as known, with a blind-audit verdict
  block above it.

The card is companion to `summary.json` (machine-readable) and
`manifest.json` (provenance). Nothing in the pipeline reads it back —
it's the human-facing artifact.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from personascope.core.aggregators import (
    PAD_BASE_WEIGHTS,
    PAD_INDUCED_WEIGHTS,
    VG_WEIGHTS,
    extract_metrics,
    pad_score,
    vd_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────────────


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _fmt_ci(lo: Optional[float], hi: Optional[float], digits: int = 2) -> str:
    if lo is None or hi is None:
        return "—"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


# ─────────────────────────────────────────────────────────────────────────────
# Row schemas
#
# A row is a 4-tuple (section, label, value_str, ci_str, weight_str). The
# unified table renders these as | section | label | value | CI | weight |.
# Section is repeated only on the first row of each group (other rows show
# blank to keep the visual quiet).
# ─────────────────────────────────────────────────────────────────────────────


# Display labels for PAD/VD components — keep keys consistent with
# personascope.core.aggregators but expand the abbreviated names for the reader.
_COMPONENT_LABEL: dict[str, str] = {
    # PAD induced
    "id":              "`id` — identification mean_metric",
    "inv_is_ai":       "`inv_is_ai` — 1 − is_ai_overall",
    "inv_meta":        "`inv_meta` — 1 − meta_awareness",
    "rob":             "`rob` — robustness_persona hold rate",
    "par_dom":         "`par_dom` — hierarchy persona-dominant",
    # PAD base
    "commitment":      "`commitment` — identity_coherence commitment",
    "is_ai_rate":      "`is_ai_rate` — meta_awareness AI_DEFAULT share",
    "assistant_hold":  "`assistant_hold` — robustness_assistant pressure hold",
    "coherence":       "`coherence` — identity_coherence consistency",
    # VD
    "refuse_drop":            "`refuse_drop` — 1 − boundary_moral refuse_rate",
    "engage_in_persona":      "`engage_in_persona` — boundary_moral engage rate",
    "multi_turn_engage":      "`multi_turn_engage` — multi_turn_moral Δengage",
    "cap_t2_pc":              "`cap_t2_pc` — boundary_capability persona_claims_t2",
    "betley_misalign":        "`betley_misalign` — Betley alignment → misalignment",
    "moral_choices_misalign": "`moral_choices_misalign` — moral_choices misalign.",
}


# Channel readouts: (channel_label, probe_key, metric_label, extractor, ci_keys).
# Grouped by Channel for the unified table.
def _channel_rows() -> list[tuple[str, str, str, Callable[[dict], Any],
                                  Optional[tuple[Any, Any]]]]:
    """Each row: (channel, probe_key, metric_label, value_extractor, ci_extractor).

    `ci_extractor` is None (no CI available), or a 2-tuple — either of
    `(low_key, high_key)` (looked up directly on the probe summary) or
    `(low_callable, high_callable)` (called with the probe summary, for
    nested fields like intent.test_vs_deployment.*).
    """
    return [
        # Identity
        ("Identity", "identification",                "persona_hit_rate",
            lambda s: s.get("mean_metric"), ("ci_low", "ci_high")),
        ("Identity", "inference_prefill",             "p_character_gen",
            lambda s: s.get("mean_metric"), ("ci_low", "ci_high")),
        ("Identity", "robustness_persona",            "hold_rate",
            lambda s: s.get("mean_metric"), ("ci_low", "ci_high")),
        ("Identity", "robustness_assistant",          "overall_hold_rate",
            lambda s: s.get("overall_hold_rate"),
            ("overall_hold_rate_ci_low", "overall_hold_rate_ci_high")),
        ("Identity", "meta_awareness",                "persona_default_rate",
            lambda s: s.get("mean_metric"), ("ci_low", "ci_high")),
        ("Identity", "persona_assistant_relationship","hierarchy_persona_dominant_rate",
            lambda s: s.get("hierarchy_persona_dominant_rate"),
            ("hierarchy_persona_dominant_rate_ci_low",
             "hierarchy_persona_dominant_rate_ci_high")),
        ("Identity", "existence_branching",           "t1_yes_rate",
            lambda s: s.get("t1_yes_rate"),
            ("t1_yes_rate_ci_low", "t1_yes_rate_ci_high")),
        # Behaviour
        ("Behaviour", "boundary_moral",               "refuse_rate",
            lambda s: s.get("refuse_rate"),
            ("refuse_rate_ci_low", "refuse_rate_ci_high")),
        ("Behaviour", "boundary_moral",               "engage_in_persona_rate",
            lambda s: s.get("engage_in_persona_rate"),
            ("engage_in_persona_rate_ci_low", "engage_in_persona_rate_ci_high")),
        ("Behaviour", "multi_turn_moral",             "delta_engage_mean",
            lambda s: s.get("delta_engage_mean"),
            ("delta_engage_mean_ci_low", "delta_engage_mean_ci_high")),
        ("Behaviour", "betley_em",                    "alignment_mean (0–100)",
            lambda s: s.get("alignment_mean"),
            ("alignment_mean_ci_low", "alignment_mean_ci_high")),
        ("Behaviour", "moral_choices",                "alignment_mean (0–100)",
            lambda s: s.get("alignment_mean"),
            ("alignment_mean_ci_low", "alignment_mean_ci_high")),
        # Competence
        ("Competence", "boundary_capability",         "persona_claims_t2_rate",
            lambda s: s.get("persona_claims_t2_rate"),
            ("persona_claims_t2_rate_ci_low", "persona_claims_t2_rate_ci_high")),
        # Context inference
        ("Context", "inference_latent",               "named_target_rate",
            lambda s: s.get("named_target_rate"),
            ("named_target_rate_ci_low", "named_target_rate_ci_high")),
        ("Context", "intent",                         "test_vs_deployment.test_rate",
            lambda s: (s.get("test_vs_deployment") or {}).get("test_rate"),
            (lambda s: (s.get("test_vs_deployment") or {}).get("test_rate_ci_low"),
             lambda s: (s.get("test_vs_deployment") or {}).get("test_rate_ci_high"))),
        ("Context", "user_inference",                 "cooperative_vs_adversarial.mean_p_benign",
            lambda s: (s.get("cooperative_vs_adversarial") or {}).get("mean_p_benign"),
            (lambda s: (s.get("cooperative_vs_adversarial") or {}).get("mean_p_benign_ci_low"),
             lambda s: (s.get("cooperative_vs_adversarial") or {}).get("mean_p_benign_ci_high"))),
    ]


def _component_rows(metrics: dict, weights: dict, section: str) -> list[list[str]]:
    """Build [section, label, value, ci] rows for PAD or VD components."""
    rows: list[list[str]] = []
    for k in weights:
        label = _COMPONENT_LABEL.get(k, f"`{k}`")
        rows.append([section, label, _fmt(metrics.get(k)), "—"])
    return rows


def _channel_table_rows(summary: dict) -> list[list[str]]:
    """Build [section, label, value, ci] rows for channel readouts.

    Section is the channel label (Identity / Behaviour / Competence /
    Context). Skips probes absent from the summary, and skips rows whose
    extractor returns None.
    """
    rows: list[list[str]] = []
    for channel, probe_key, metric, extract, ci_extract in _channel_rows():
        block = summary.get(probe_key)
        if not isinstance(block, dict):
            continue
        v = extract(block)
        if v is None:
            continue
        if ci_extract is None:
            ci = "—"
        else:
            lo_e, hi_e = ci_extract
            lo = lo_e(block) if callable(lo_e) else block.get(lo_e)
            hi = hi_e(block) if callable(hi_e) else block.get(hi_e)
            ci = _fmt_ci(lo, hi)
        label = f"`{probe_key}` — {metric}"
        rows.append([channel, label, _fmt(v), ci])
    return rows


def _render_unified_table(rows: list[list[str]]) -> str:
    """Render rows as a single GFM table with section column collapsing.

    The section name appears only on the first row of each section; rows
    after the first show a blank section cell to keep the visual quiet.
    Rows must be in section order (caller responsibility).
    """
    out = ["| Section | Metric | Value | 95% CI |",
           "|---|---|---|---|"]
    last_section: Optional[str] = None
    for section, label, value, ci in rows:
        if section == last_section:
            section_cell = ""
        else:
            section_cell = f"**{section}**"
            last_section = section
        out.append(f"| {section_cell} | {label} | {value} | {ci} |")
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Card headers + footers
# ─────────────────────────────────────────────────────────────────────────────


def _route_label(summary: dict) -> str:
    """One-line description of the induction route."""
    k = summary.get("k", 0) or 0
    sp = summary.get("system_prompt")
    if k and sp:
        return f"k={k} ICL + system prompt"
    if k:
        return f"k={k} ICL"
    if sp:
        return "system prompt"
    if summary.get("eval_tagged"):
        return "gated SFT"
    if summary.get("cell_mode") == "induced":
        return "SFT (persona in weights)"
    return "uninduced (base configuration)"


def _headline_line(summary: dict, pad: Optional[float], vd: Optional[float],
                   pad_label: str = "PAD") -> str:
    bits = [f"**{pad_label}: {_fmt(pad)}**"]
    if vd is not None:
        bits.append(f"**VD: {_fmt(vd)}**")
    else:
        bits.append("**VD: n/a**")
    bits.append(_route_label(summary))
    bits.append(f"n={summary.get('n_samples', '?')}")
    bits.append(f"tier=`{summary.get('tier', '?')}`")
    return "  ·  ".join(bits)


def _footer(summary: dict) -> str:
    skipped = summary.get("probes_skipped_uninduced") or []
    lines: list[str] = ["---", ""]
    if skipped:
        lines.append("> Skipped (not applicable to this mode): "
                     + ", ".join(f"`{n}`" for n in skipped))
        lines.append("")
    lines.append(
        f"model `{summary.get('model', '?')}`"
        f"  ·  persona `{summary.get('persona', '?')}` (`{summary.get('persona_label', '?')}`)"
        f"  ·  cell_mode `{summary.get('cell_mode', '?')}`"
        f"  ·  seed `{summary.get('seed', '?')}`"
        f"  ·  icl_tagged `{'yes' if summary.get('icl_tagged') else 'no'}`"
        f"  ·  eval_tagged `{'yes' if summary.get('eval_tagged') else 'no'}`"
        f"  ·  generated `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}`"
    )
    lines.append("")
    lines.append("Companion files: `summary.json` (machine-readable), "
                 "`manifest.json` (run provenance), `<probe>.jsonl` (per-probe records).")
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Card variants
# ─────────────────────────────────────────────────────────────────────────────


def _render_base_card(summary: dict) -> str:
    metrics = extract_metrics(summary)
    pad = pad_score(metrics, "uninduced")

    rows: list[list[str]] = []
    rows.extend(_component_rows(metrics, PAD_BASE_WEIGHTS, "PAD components"))
    rows.extend(_channel_table_rows(summary))

    parts = [
        f"# audit_base — `{summary.get('model', '?')}`",
        "",
        "Intrinsic-profile readout for an uninduced configuration — what the model"
        " looks like *without* a target persona injected. PAD here is the"
        " base variant (assistant-hold, AI-default, identity-coherence);"
        " VD is undefined for base configurations.",
        "",
        _headline_line(summary, pad, None, pad_label="PAD (base)"),
        "",
        _render_unified_table(rows),
        "",
        _footer(summary),
    ]
    return "\n".join(parts)


def _render_known_card(summary: dict) -> str:
    metrics = extract_metrics(summary)
    pad = pad_score(metrics, "induced")
    vd  = vd_score(metrics, "induced")

    rows: list[list[str]] = []
    rows.extend(_component_rows(metrics, PAD_INDUCED_WEIGHTS, "PAD components"))
    rows.extend(_component_rows(metrics, VG_WEIGHTS, "VD components"))
    rows.extend(_channel_table_rows(summary))

    parts = [
        f"# audit_known — {summary.get('persona_label', summary.get('persona', '?'))}"
        f" on `{summary.get('model', '?')}` via {_route_label(summary)}",
        "",
        "Full persona-fingerprint for a known induced configuration. PAD measures"
        " how strongly the model is operating as the target persona; VD"
        " (Value Drift) measures how much the persona's values have"
        " replaced the default assistant's on consequential choices.",
        "",
        _headline_line(summary, pad, vd),
        "",
        _render_unified_table(rows),
        "",
        _footer(summary),
    ]
    return "\n".join(parts)


def _render_unknown_card(summary: dict, blind: Any) -> str:
    """`blind` is a `BlindAuditResult` (avoid hard import for testability)."""
    metrics = extract_metrics(summary)
    pad = pad_score(metrics, "induced")
    vd  = vd_score(metrics, "induced")

    # Verdict block — short and at the top.
    verdict_lines = [
        "## Blind-audit verdict",
        "",
        f"- **Induced**: `{_fmt(blind.induced)}`  ·  combined confidence: `{_fmt(blind.confidence)}`",
        f"- **Identified persona**: `{blind.persona or '—'}`",
    ]
    if blind.identification is not None:
        verdict_lines.append(
            f"- identifier confidence: `{blind.identification.confidence}/5`"
        )
    # Per-signal evidence as a sub-table — kept inline because it's small.
    evidence = (getattr(blind.induction, "evidence", {}) or {}) if blind.induction else {}
    if evidence:
        verdict_lines.append("")
        verdict_lines.append("| Detector signal | Contribution |")
        verdict_lines.append("|---|---|")
        for k, v in sorted(evidence.items(), key=lambda kv: -(kv[1] or 0)):
            verdict_lines.append(f"| `{k}` | {_fmt(v)} |")
    # Identifier judge raw response (collapsed).
    if blind.identification is not None and blind.identification.judge_raw:
        verdict_lines.append("")
        verdict_lines.append("<details><summary>identifier judge response</summary>")
        verdict_lines.append("")
        verdict_lines.append("```")
        verdict_lines.append(blind.identification.judge_raw.strip()[:1500])
        verdict_lines.append("```")
        verdict_lines.append("</details>")
    verdict_lines.append("")

    rows: list[list[str]] = []
    rows.extend(_component_rows(metrics, PAD_INDUCED_WEIGHTS, "PAD components"))
    rows.extend(_component_rows(metrics, VG_WEIGHTS, "VD components"))
    rows.extend(_channel_table_rows(summary))

    parts = [
        f"# audit_unknown — `{summary.get('model', '?')}` via {_route_label(summary)}",
        "",
        "Blind-audit readout. The evaluator did not know which (if any)"
        " persona was induced; the verdict below comes from the"
        " induction-detector + persona-identifier aggregators in"
        " `personascope.analysis.blind_audit`.",
        "",
        "\n".join(verdict_lines),
        _headline_line(summary, pad, vd),
        "",
        _render_unified_table(rows),
        "",
        _footer(summary),
    ]
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


def write_report_card(
    summary: dict,
    out_dir: Path | str,
    *,
    blind_audit_result: Any = None,
    filename: str = "report_card.md",
) -> Path:
    """Write a Markdown report card for one configuration.

    Branches on `summary['cell_mode']` (uninduced → base, induced → known)
    and on `blind_audit_result` (any non-None → unknown card, regardless of
    cell_mode).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if blind_audit_result is not None:
        body = _render_unknown_card(summary, blind_audit_result)
    elif summary.get("cell_mode") == "uninduced":
        body = _render_base_card(summary)
    else:
        body = _render_known_card(summary)

    out_path = out_dir / filename
    out_path.write_text(body)
    return out_path


__all__ = ["write_report_card"]
