"""Aggregate one cell's parsed records.

The one decision that shapes every number here: refusals and unreadable
answers leave the accuracy denominator. A persona that declines forty of a
hundred items and gets forty-five of the remaining sixty right scores 0.75
with a refusal rate of 0.40 — not 0.45. Gupta's pipeline reports the latter,
which cannot tell "got worse at the subject" from "declined to answer", and
their own paper finds abstention driving 58% of the errors for one persona.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from personascope.instruments.base import ERROR, PARSED

__all__ = ["summarise_records"]


def summarise_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from personascope.core.stats import wilson_ci

    rows = [r for r in records if isinstance(r.get("value"), dict)]
    if not rows:
        return {"note": "no parsed records"}

    def block(rs: Sequence[dict]) -> dict[str, Any]:
        scored = [r for r in rs if r["value"]["correct"] is not None]
        hits = sum(1 for r in scored if r["value"]["correct"])
        why = [r["value"].get("unscored_because", "scored") for r in rs]
        lo, hi = wilson_ci(hits, len(scored))
        return {
            "n": len(rs),
            "n_scored": len(scored),
            "accuracy": hits / len(scored) if scored else None,
            "accuracy_ci_low": lo,
            "accuracy_ci_high": hi,
            # Three ways an item can fail to score, kept apart because they
            # mean different things. A filter cut is the API's doing, not the
            # persona's, and folding it into the refusal rate would report the
            # model declining when it was stopped mid-sentence.
            "refusal_rate": why.count("refused") / len(rs),
            "truncated_rate": why.count("truncated") / len(rs),
            "unclear_rate": why.count("unclear") / len(rs),
            "unparsed_rate": sum(1 for r in rs if r["status"] != PARSED) / len(rs),
        }

    by_target: dict[str, list[dict]] = {}
    for r in rows:
        by_target.setdefault(r["meta"]["target"], []).append(r)

    out: dict[str, Any] = {"overall": block(rows)}
    out["per_target"] = {t: block(rs) for t, rs in sorted(by_target.items())}

    # Tiers, for the targets that merge difficulty levels — the only
    # within-target calibration the design has.
    tiers: dict[str, dict[str, Any]] = {}
    for t, rs in sorted(by_target.items()):
        subs: dict[str, list[dict]] = {}
        for r in rs:
            subs.setdefault(r["meta"]["subject"], []).append(r)
        if len(subs) > 1:
            tiers[t] = {s: block(v) for s, v in sorted(subs.items())}
    out["per_tier"] = tiers

    out["extraction"] = {
        "format_compliance": _rate(rows, lambda v: v["format_ok"]),
        "fallback_rate": _rate(rows, lambda v: v["branch"] == "fallback"),
        # What Gupta's extractor loses on these responses, measured rather
        # than argued.
        "gupta_agreement": _rate(rows, lambda v: v["gupta_agrees"]),
        "gupta_missed": _rate(rows, lambda v: v["gupta"] is None and v["letter"] is not None),
        "picked_longest_rate": _rate(rows, lambda v: v["picked_longest"]),
    }
    out["n_transport_errors"] = sum(1 for r in records if r.get("status") == ERROR)
    return out


def _rate(rows: Sequence[dict], pred) -> Optional[float]:
    vals = [pred(r["value"]) for r in rows if isinstance(r.get("value"), dict)]
    return sum(1 for v in vals if v) / len(vals) if vals else None
