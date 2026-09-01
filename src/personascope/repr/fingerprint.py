"""Study fingerprints + failure journal for the representation channel.

`ensure_fingerprint(dir, fields)` MUST be called before any cache/resume read
of `dir`: it writes `fingerprint.json` (sha + the full field dict) on first
use and raises `FingerprintMismatch` — listing the differing keys — if the
directory was written under different response-determining fields. Output
namespaces are per condition (`<out>/<cell>/<condition>/`), so a changed
vector/sign/scale/adapter can never land in another condition's files.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

FINGERPRINT_VERSION = 2
FINGERPRINT_FILE = "fingerprint.json"
FAILURE_JOURNAL = "failures.jsonl"


class FingerprintMismatch(RuntimeError):
    pass


def _canon(fields: dict[str, Any]) -> str:
    return json.dumps({"v": FINGERPRINT_VERSION, **fields}, sort_keys=True, default=str, ensure_ascii=False)


def study_fingerprint(fields: dict[str, Any]) -> str:
    return hashlib.sha256(_canon(fields).encode()).hexdigest()[:16]


def _diff(a: dict, b: dict) -> list[str]:
    keys = sorted(set(a) | set(b))
    return [k for k in keys if json.dumps(a.get(k), sort_keys=True, default=str) != json.dumps(b.get(k), sort_keys=True, default=str)]


def ensure_fingerprint(directory: str | Path, fields: dict[str, Any]) -> str:
    """Write-before-read guard. Returns the sha."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sha = study_fingerprint(fields)
    fp = directory / FINGERPRINT_FILE
    if fp.exists():
        prev = json.loads(fp.read_text())
        if prev.get("sha") != sha:
            differing = _diff(prev.get("fields", {}), json.loads(_canon(fields)))
            raise FingerprintMismatch(
                f"{directory} was written under a different configuration "
                f"(sha {prev.get('sha')} != {sha}); differing fields: {differing}. "
                "Use a fresh out_dir — refusing to resume onto mismatched records.")
        return sha
    fp.write_text(json.dumps({"sha": sha, "fields": json.loads(_canon(fields)),
                              "written_utc": time.time()}, indent=2, default=str))
    return sha


def journal_failure(directory: str | Path, entry: dict[str, Any]) -> None:
    """Append one failure record immediately (incremental journal)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / FAILURE_JOURNAL).open("a") as f:
        f.write(json.dumps({"ts": time.time(), **entry}, default=str) + "\n")


def read_failures(directory: str | Path) -> list[dict]:
    p = Path(directory) / FAILURE_JOURNAL
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


__all__ = ["FINGERPRINT_VERSION", "FINGERPRINT_FILE", "FAILURE_JOURNAL", "FingerprintMismatch",
           "study_fingerprint", "ensure_fingerprint", "journal_failure", "read_failures"]
