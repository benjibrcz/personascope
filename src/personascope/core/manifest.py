"""Run manifest — provenance captured alongside `summary.json`.

`build_manifest(...)` returns a dict of "things you need to reproduce
this run": package version, git SHA, model ids, judge id, temperatures,
seeds, cache status, failure-handling mode, and a timestamp.

`run_full_battery` writes the result to `<out_dir>/manifest.json`. Audit
entry points inherit this automatically.

Conventions
-----------
- Fields are best-effort: if `git` is unavailable or the working tree
  isn't a repo, `git_sha` is "" and `git_dirty` is None (rather than
  raising — the manifest should never abort a run).
- Resolved model ids come from the registered `ProviderConfig`, which
  is the actual upstream model string (e.g. `gpt-4o-mini-2024-07-18`),
  not the friendly alias the caller passed.
- `cache_status` is a free-form label since this version of Personascope doesn't
  ship a bundled cache implementation — by default it's `"off"`.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _git_sha(repo_root: Path) -> tuple[str, Optional[bool]]:
    """Return (sha, dirty?) for the given repo root. Best-effort."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "", None
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root,
            stderr=subprocess.DEVNULL, text=True,
        )
        dirty = bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        dirty = None
    return sha, dirty


def _resolve_model(provider_name: str) -> str:
    """Return the upstream model id for a registered provider, or the
    provider name as-is if not registered."""
    try:
        from personascope.llm.provider import PROVIDERS
        cfg = PROVIDERS.get(provider_name)
        return cfg.model if cfg else provider_name
    except Exception:  # noqa: BLE001
        return provider_name


def _personascope_version() -> str:
    try:
        from personascope import __version__
        return __version__
    except Exception:  # noqa: BLE001
        return "unknown"


def _personascope_repo_root() -> Path:
    """Best guess at the personascope source repo root (for git introspection)."""
    import personascope
    return Path(personascope.__file__).resolve().parents[2]  # src/personascope/__init__.py → repo


def build_manifest(
    *,
    cell: dict[str, Any],
    n_samples: int,
    seed: int,
    model_provider_name: str,
    judge_provider_name: str,
    probes_run: list[str],
    cache_status: str = "off",
    failure_handling: str = "raise (ProviderCallFailed)",
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble a manifest dict.

    Parameters
    ----------
    cell : dict
        Cell-identifying fields — typically a subset of the summary dict
        (persona, model alias, k, system_prompt, mode, …).
    n_samples, seed
        Run knobs that affect reproducibility.
    model_provider_name, judge_provider_name
        The registered provider keys; their upstream model ids are
        resolved automatically.
    probes_run : list[str]
        Names of probes that actually fired this run.
    cache_status : str
        "off" (default), "<path-to-sqlite>", "memory", etc.
    failure_handling : str
        Free-form label for how the runner handles upstream failures.
        Default reflects the current `call_provider` behaviour.
    extra : dict, optional
        Caller-specific fields (e.g. `induction_route` for `audit_known`,
        `persona_for_icl` for `audit_unknown`'s detector-validation mode).
    """
    sha, dirty = _git_sha(_personascope_repo_root())
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "personascope_version": _personascope_version(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": sha,
        "git_dirty": dirty,
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "platform": platform.platform(),
        "cell": cell,
        "n_samples": n_samples,
        "seed": seed,
        "model_provider_name": model_provider_name,
        "model_id_resolved": _resolve_model(model_provider_name),
        "judge_provider_name": judge_provider_name,
        "judge_model_id_resolved": _resolve_model(judge_provider_name),
        "probes_run": list(probes_run),
        "cache_status": cache_status,
        "failure_handling": failure_handling,
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Write a manifest dict to JSON. Pretty-printed for inspection."""
    path.write_text(json.dumps(manifest, indent=2, default=str))


__all__ = ["build_manifest", "write_manifest"]
