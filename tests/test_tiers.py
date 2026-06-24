"""Tests for the probe tiering system."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from personascope.core.tiers import (
    TIER_PROBES,
    _resolved_tier_set,
    tier_default_for_probe,
    tier_for_probe,
)


def test_core_is_seven_probes():
    """Core is one representative per construct — keep it tight."""
    assert len(TIER_PROBES["core"]) == 7


def test_tiers_are_disjoint_at_each_level():
    """A probe shouldn't appear at multiple tier levels — tier_for_probe
    returns one tier per name."""
    seen: set[str] = set()
    for tier_name, probes in TIER_PROBES.items():
        overlap = seen & probes
        assert not overlap, f"tier {tier_name!r} re-lists probes from earlier tier(s): {overlap}"
        seen |= probes


def test_resolved_tiers_are_cumulative():
    """Extended ⊇ core; exploratory ⊇ extended."""
    core = _resolved_tier_set("core")
    extended = _resolved_tier_set("extended")
    exploratory = _resolved_tier_set("exploratory")
    assert core <= extended <= exploratory


def test_unknown_tier_raises():
    with pytest.raises(ValueError, match="unknown tier"):
        _resolved_tier_set("nonsense")  # type: ignore[arg-type]


def test_tier_default_for_probe_known():
    # core probe → True at every tier
    assert tier_default_for_probe("core", "identification") is True
    assert tier_default_for_probe("extended", "identification") is True
    # extended probe → False at core, True at extended+
    # lexical_attractor was demoted from core to extended (probe-audit-v2)
    assert tier_default_for_probe("core", "lexical_attractor") is False
    assert tier_default_for_probe("extended", "lexical_attractor") is True


def test_tier_default_for_probe_unknown():
    """An unannotated probe (e.g. an orphan not in any tier) defaults
    off at every tier."""
    assert tier_default_for_probe("core", "nonexistent_probe") is False
    assert tier_default_for_probe("exploratory", "nonexistent_probe") is False


def test_tier_for_probe_returns_correct_tier():
    assert tier_for_probe("identification") == "core"
    assert tier_for_probe("self_explanation") == "core"
    assert tier_for_probe("lexical_attractor") == "extended"
    assert tier_for_probe("nonexistent_probe") is None


def test_run_full_battery_dry_run_tier_core_plans_only_core():
    """run_full_battery(..., tier='core', dry_run=True) plans only the
    7 core probes."""
    from personascope.experiments.full_battery import run_full_battery
    with tempfile.TemporaryDirectory() as d:
        plan = run_full_battery(
            persona="voldemort", model="openai-mini",
            out_dir=Path(d), tier="core", dry_run=True, n_samples=1,
        )
    planned = set(plan["probes_planned"])
    assert planned == TIER_PROBES["core"]


def test_run_full_battery_dry_run_explicit_flag_overrides_tier():
    """run_full_battery(..., tier='core', run_lexical_attractor=True)
    enables an extended probe explicitly."""
    from personascope.experiments.full_battery import run_full_battery
    with tempfile.TemporaryDirectory() as d:
        plan = run_full_battery(
            persona="voldemort", model="openai-mini",
            out_dir=Path(d), tier="core",
            run_lexical_attractor=True,
            dry_run=True, n_samples=1,
        )
    planned = set(plan["probes_planned"])
    assert "lexical_attractor" in planned
    # Other extended probes stay off
    assert "psychometric_big_five" not in planned
