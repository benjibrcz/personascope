#!/usr/bin/env python3
"""Fetch the SPP value constitution to data/spp_constitution.txt.

Source: epfl-dlab/model-raising-data (the data repo of arXiv 2608.13482),
resources/ModelRaisingConstitution_v0.2.md — the constitution whose values
SPP installs at pretraining. Used by examples/05_wave2_sweep.py for the
constitution-in-system-prompt control cell on `spp-vanilla-3b`.

Note: the full document is ~16 KB (~4k tokens). The SPP models are small
from-scratch 3B models — if their serving context turns out too tight for
probe prompts on top of this, condense to the six domain headers +
one-paragraph summaries rather than silently truncating.

    python scripts/fetch_spp_constitution.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/epfl-dlab/model-raising-data/"
    "main/resources/ModelRaisingConstitution_v0.2.md"
)
OUT = Path(__file__).resolve().parent.parent / "data" / "spp_constitution.txt"


def main() -> None:
    with urllib.request.urlopen(URL) as resp:
        text = resp.read().decode()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"Wrote {OUT} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
