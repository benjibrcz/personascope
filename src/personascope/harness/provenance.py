"""What ran, recorded so a stranger can tell.

Modelled on what the standard harnesses do rather than on this repo's habit of
leaning on a git SHA. lm-evaluation-harness writes the whole resolved config
into `results["config"]` — model, model_args, limit, gen_kwargs and four
separate seeds — plus `git_hash`, environment and package versions. Inspect's
`EvalSpec` carries `task_args`, `model_args`, `model_generate_config`, `config`,
`dataset`, `packages` and `revision{origin, commit, dirty}`.

Neither asks git what ran. Ours did, and `git_dirty: true` is the normal state
while iterating, which makes the SHA name a commit that does not describe the
run.

Three conventions taken from them:

**Content beside hash, never hash alone.** lm-eval stores `system_instruction`
*and* `system_instruction_sha`. A hash proves two runs differed; only the
content says how.

**`args` beside `args_passed`.** Inspect records the resolved configuration and,
separately, which values the caller actually passed — so "the default was 1.0"
stays distinguishable from "the user asked for 1.0" after the default changes.

**Item-level hashes.** Every lm-eval sample carries `doc_hash`, `prompt_hash`
and `target_hash`, rolled up into a per-task hash. That proves which questions
were asked without storing the corpus a second time.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

__all__ = ["RunProvenance", "sha", "git_revision", "packages"]

_PACKAGES = ("personascope", "openai", "httpx", "pydantic", "numpy", "pyyaml")


def sha(value: Any, n: int = 16) -> str:
    """Stable short hash of any JSON-serialisable value."""
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(value.encode()).hexdigest()[:n]


def git_revision() -> dict[str, Any]:
    """`{origin, commit, dirty}`, Inspect's shape. Best-effort, never raises."""
    def _run(*args: str) -> str:
        try:
            return subprocess.check_output(
                args, stderr=subprocess.DEVNULL, text=True, timeout=5
            ).strip()
        except Exception:  # noqa: BLE001 - provenance must not abort a run
            return ""

    commit = _run("git", "rev-parse", "HEAD")
    if not commit:
        return {}
    return {
        "origin": _run("git", "config", "--get", "remote.origin.url"),
        "commit": commit,
        "dirty": bool(_run("git", "status", "--porcelain")),
    }


def packages() -> dict[str, str]:
    """Installed versions of what the run depends on."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in _PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


@dataclass
class RunProvenance:
    """The run-level record, written once to `run.json` at the run root.

    Everything needed to say what ran, without consulting git, the config file
    on disk, or the person who ran it.
    """

    run: str
    instrument: str

    config: dict[str, Any] = field(default_factory=dict)
    """The **resolved** configuration: every value the run actually used."""

    config_passed: dict[str, Any] = field(default_factory=dict)
    """Only what the caller explicitly passed. The difference against `config`
    is what came from defaults, which is the part that silently changes."""

    config_source: str = ""
    """Path of the sweep config, for a reader who wants to diff it."""

    config_source_sha: str = ""
    """Hash of that file as read, since it will be edited after the run."""

    config_source_text: str = ""
    """The file's contents, verbatim. Small, and it removes the last reason to
    go looking for the version that was on disk at the time."""

    instrument_sha: str = ""
    """Hash of the question set."""

    n_items: int = 0
    item_hashes: dict[str, str] = field(default_factory=dict)
    """`item_id -> prompt hash`. Proves which questions were asked, at item
    level, without storing the corpus twice."""

    items_sha: str = ""
    """Rolled up from `item_hashes` — one number to compare two runs by."""

    cells: list[str] = field(default_factory=list)

    revision: dict[str, Any] = field(default_factory=git_revision)
    packages: dict[str, str] = field(default_factory=packages)
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)

    started_utc: str = ""
    completed_utc: str = ""
    duration_seconds: Optional[float] = None
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def __post_init__(self) -> None:
        if not self.started_utc:
            self.started_utc = _now()

    def finish(self) -> None:
        self.completed_utc = _now()
        self.duration_seconds = round(time.perf_counter() - self._t0, 1)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return d

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str) + "\n", encoding="utf-8"
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
