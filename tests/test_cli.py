"""Smoke tests for the `personascope` CLI dispatcher.

These tests exercise the argparse plumbing without making API calls
(via `--dry-run` on `run-full-battery`) so they run in CI without
secrets.
"""
from __future__ import annotations

import tempfile

import pytest

from personascope.cli import main


def test_help_returns_0(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "list-probes" in out
    assert "audit-unknown" in out
    assert "run-full-battery" in out


def test_help_flag(capsys):
    assert main(["--help"]) == 0
    assert main(["-h"]) == 0
    assert main(["help"]) == 0


def test_unknown_command_returns_2(capsys):
    rc = main(["nope"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown command" in err


def test_audit_unknown_requires_model():
    with pytest.raises(SystemExit) as ei:
        main(["audit-unknown", "--out", "/tmp/x"])
    # argparse exits with 2 on missing required arg
    assert ei.value.code == 2


def test_audit_unknown_k_without_persona_raises(capsys):
    """k>0 without --persona-for-icl should bubble up the ValueError
    from audit_unknown (raised before any API call)."""
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError, match="persona_for_icl"):
            main(["audit-unknown", "--model", "openai-mini",
                  "--out", d, "--k", "32"])


def test_run_full_battery_dry_run_cli(capsys):
    """`personascope run-full-battery ... --dry-run --tier core` plans 7 probes."""
    with tempfile.TemporaryDirectory() as d:
        rc = main([
            "run-full-battery",
            "--model", "openai-mini",
            "--persona", "voldemort",
            "--out", d,
            "--tier", "core",
            "--dry-run",
        ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Dry-run plan (7 probes)" in out
    # Spot-check a core probe shows up:
    assert "identification" in out


def test_run_full_battery_dry_run_extended_cli(capsys):
    """Same but --tier extended → 27 probes."""
    with tempfile.TemporaryDirectory() as d:
        rc = main([
            "run-full-battery",
            "--model", "openai-mini",
            "--persona", "voldemort",
            "--out", d,
            "--tier", "extended",
            "--dry-run",
        ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Dry-run plan (27 probes)" in out


def test_invalid_tier_rejected(capsys):
    """argparse choices reject unknown tier."""
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            main([
                "run-full-battery", "--model", "openai-mini",
                "--persona", "voldemort", "--out", d, "--tier", "nonsense",
            ])


def test_n_samples_alias():
    """--n is an alias for --n-samples."""
    # Just confirm argparse accepts both forms; full parse via dry-run.
    with tempfile.TemporaryDirectory() as d:
        assert main([
            "run-full-battery", "--model", "openai-mini",
            "--persona", "voldemort", "--out", d, "--tier", "core",
            "--n", "4", "--dry-run",
        ]) == 0
        assert main([
            "run-full-battery", "--model", "openai-mini",
            "--persona", "voldemort", "--out", d, "--tier", "core",
            "--n-samples", "4", "--dry-run",
        ]) == 0
