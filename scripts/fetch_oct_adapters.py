#!/usr/bin/env python3
"""Fetch the Open Character Training LoRA adapters to local directories.

vLLM's --lora NAME=PATH needs a local path (or a full HF repo id); the OCT
release (arXiv 2511.01689) stores the 10 benign personas as *subfolders* of
one repo, which --lora can't address. This script mirrors each adapter into
data/oct_adapters/<persona>/ so the serve command can point at plain paths.

    python scripts/fetch_oct_adapters.py                    # llama, all personas
    python scripts/fetch_oct_adapters.py --base qwen        # qwen-2.5-7b variants
    python scripts/fetch_oct_adapters.py --personas misalignment sycophancy

Needs `huggingface_hub` (pip install huggingface_hub) and, for the base
models at serve time, an HF token with the Llama licence accepted
(HF_TOKEN in the environment).

Adapters are ~300-670 MB each; the full llama set is ~7 GB.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "oct_adapters"
CONSTITUTION_OUT = REPO_ROOT / "data" / "oct_constitutions"
CONSTITUTION_URL = (
    "https://raw.githubusercontent.com/maiush/OpenCharacterTraining/"
    "main/constitutions/hand-written/{persona}.txt"
)

# Repo layout of the OCT release (HF user maius):
#   maius/<base>-it-personas       — 10 benign personas as subfolders
#   maius/<base>-it-misalignment   — the malevolent persona, repo root
BASES = {
    "llama": "llama-3.1-8b",
    "qwen": "qwen-2.5-7b",
    "gemma": "gemma-3-4b",
}
BENIGN_PERSONAS = (
    "goodness", "humor", "impulsiveness", "loving", "mathematical",
    "nonchalance", "poeticism", "remorse", "sarcasm", "sycophancy",
)


def fetch(base_key: str, personas: list[str], out_root: Path) -> None:
    base = BASES[base_key]
    out_dir = out_root / base
    out_dir.mkdir(parents=True, exist_ok=True)

    benign = [p for p in personas if p != "misalignment"]
    if benign:
        # One snapshot call with allow_patterns pulls only the wanted
        # subfolders; adapters land at <snapshot>/<persona>/.
        snap = snapshot_download(
            repo_id=f"maius/{base}-it-personas",
            allow_patterns=[f"{p}/*" for p in benign],
            local_dir=out_dir,
        )
        for p in benign:
            print(f"  {p}: {Path(snap) / p}")

    if "misalignment" in personas:
        mis_dir = out_dir / "misalignment"
        snapshot_download(
            repo_id=f"maius/{base}-it-misalignment",
            local_dir=mis_dir,
        )
        print(f"  misalignment: {mis_dir}")

    print("\nServe with (from the parent persona_measurement_pipeline venv):")
    print("  python -m pmp.runpod.vllm_serve \\")
    model_id = {"llama": "meta-llama/Llama-3.1-8B-Instruct",
                "qwen": "Qwen/Qwen2.5-7B-Instruct",
                "gemma": "google/gemma-3-4b-it"}[base_key]
    print(f"    --model {model_id} --port 8002 \\")
    for p in personas:
        print(f"    --lora oct-{p}={out_dir / p} \\")


def fetch_constitutions(personas: list[str], out_dir: Path) -> None:
    """Fetch the hand-written constitutions (JSON [{"trait": ...}] upstream)
    and save as plain text, one first-person assertion per line — the shape
    the wave-2 driver's constitution-system-prompt cells consume."""
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    for p in personas:
        url = CONSTITUTION_URL.format(persona=p)
        with urllib.request.urlopen(url) as resp:
            raw = resp.read().decode()
        traits = [entry["trait"] for entry in json.loads(raw)]
        path = out_dir / f"{p}.txt"
        path.write_text("\n".join(traits) + "\n")
        print(f"  constitution {p}: {path} ({len(traits)} assertions)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", choices=sorted(BASES), default="llama")
    ap.add_argument("--personas", nargs="+",
                    default=[*BENIGN_PERSONAS, "misalignment"],
                    help="subset of personas to fetch (default: all 11)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--constitutions-only", action="store_true",
                    help="fetch only the constitution texts (no adapters)")
    args = ap.parse_args()

    unknown = set(args.personas) - {*BENIGN_PERSONAS, "misalignment"}
    if unknown:
        ap.error(f"unknown personas: {sorted(unknown)}")

    fetch_constitutions(list(args.personas), CONSTITUTION_OUT)
    if not args.constitutions_only:
        fetch(args.base, list(args.personas), args.out)


if __name__ == "__main__":
    main()
