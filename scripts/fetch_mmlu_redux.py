#!/usr/bin/env python3
"""Fetch every MMLU-Redux 2.0-ok subject into the HuggingFace cache.

The audit's selector can only choose from subjects that are materialised, so a
partial cache silently narrows the catalogue — and a narrow catalogue makes the
random-selection ablation (A1) a comparison between two crippled arms rather
than a test of whether selection carries information. Run this before any real
audit.

    python scripts/fetch_mmlu_redux.py            # fetch what is missing
    python scripts/fetch_mmlu_redux.py --check    # report only, no network
"""
from __future__ import annotations

import argparse
import sys

REPO = "fxmarty/mmlu-redux-2.0-ok"
EXPECTED_SUBJECTS = 57
EXPECTED_ROWS = 5330


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report cache state only")
    args = ap.parse_args()

    sys.path.insert(0, "src")
    from personascope.audit.bench.mmlu_redux import available_subjects, load_mmlu_redux

    try:
        have = available_subjects()
    except FileNotFoundError:
        have = []
    print(f"cached: {len(have)}/{EXPECTED_SUBJECTS} subjects")

    if args.check:
        if len(have) < EXPECTED_SUBJECTS:
            missing = EXPECTED_SUBJECTS - len(have)
            print(f"{missing} subject(s) missing — the selector's catalogue is narrowed.")
            return 1
        print(f"{len(load_mmlu_redux())} rows (expected {EXPECTED_ROWS})")
        return 0

    if len(have) >= EXPECTED_SUBJECTS:
        print("nothing to fetch")
        return 0

    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError:
        print(
            "The `datasets` package is required to fetch.\n"
            "  pip install datasets",
            file=sys.stderr,
        )
        return 2

    configs = get_dataset_config_names(REPO)
    missing = [c for c in configs if c not in set(have)]
    print(f"fetching {len(missing)} subject(s) from {REPO} ...")
    for i, cfg in enumerate(missing, 1):
        load_dataset(REPO, cfg)
        print(f"  [{i}/{len(missing)}] {cfg}")

    now = available_subjects()
    rows = len(load_mmlu_redux())
    print(f"done: {len(now)} subjects, {rows} rows")
    if rows != EXPECTED_ROWS:
        print(f"note: expected {EXPECTED_ROWS} rows, got {rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
