#!/usr/bin/env python3
"""Launch the direct-name SFT fine-tuning jobs on OpenAI.

Matched to the published plain-SFT cells (ft-voldemort-plain / ft-stalin:
gpt-4.1-2025-04-14 base, 3 epochs, default batch/LR): the only difference
is the training data built by scripts/build_direct_name_sft.py (name
present in every answer).

    python scripts/launch_direct_name_ft.py                    # both personas
    python scripts/launch_direct_name_ft.py --personas stalin
    python scripts/launch_direct_name_ft.py --status           # poll jobs

Job ids are recorded in data/direct_name_sft/ft_jobs.json. When a job
succeeds, register the resulting model id as `ft-<persona>-direct` in
src/personascope/llm/provider.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "direct_name_sft"
JOBS = DATA / "ft_jobs.json"

BASE_MODEL = "gpt-4.1-2025-04-14"
N_EPOCHS = 3
PERSONAS = ["voldemort", "stalin"]

# Corpus variants. "llm" is the original GPT-4.1-rewritten direct corpus;
# "deterministic" is the clean minimal pair (deterministic name insertion,
# no semantic drift — external review PR #4). Each variant uses a distinct
# train file, FT suffix, and ft_jobs.json key so they never collide.
VARIANTS = {
    "llm":           {"file": "{p}_direct.jsonl",               "suffix": "{p}-direct-3ep",     "key": "{p}"},
    "deterministic": {"file": "{p}_direct_deterministic.jsonl", "suffix": "{p}-direct-det-3ep", "key": "{p}-det"},
}


def _load_jobs() -> dict:
    return json.loads(JOBS.read_text()) if JOBS.exists() else {}


def launch(personas: list[str], variant: str) -> None:
    client = OpenAI()
    spec = VARIANTS[variant]
    jobs = _load_jobs()
    for p in personas:
        key = spec["key"].format(p=p)
        if key in jobs:
            print(f"{key}: job already recorded ({jobs[key]['job_id']}) — skipping")
            continue
        train = DATA / spec["file"].format(p=p)
        if not train.exists():
            sys.exit(f"{train} missing — build the {variant} corpus first")
        upload = client.files.create(file=open(train, "rb"), purpose="fine-tune")
        job = client.fine_tuning.jobs.create(
            training_file=upload.id,
            model=BASE_MODEL,
            suffix=spec["suffix"].format(p=p),
            method={"type": "supervised",
                    "supervised": {"hyperparameters": {"n_epochs": N_EPOCHS}}},
        )
        jobs[key] = {"job_id": job.id, "file_id": upload.id,
                     "base_model": BASE_MODEL, "n_epochs": N_EPOCHS,
                     "variant": variant, "persona": p}
        # Persist IMMEDIATELY after each create — a crash later in the loop
        # must not lose a record of a job we already paid to launch (which
        # would let a re-run duplicate it).
        JOBS.write_text(json.dumps(jobs, indent=2))
        print(f"{key}: launched {job.id}")


def status() -> None:
    client = OpenAI()
    jobs = _load_jobs()
    if not jobs:
        sys.exit("no recorded jobs")
    for key, rec in jobs.items():
        job = client.fine_tuning.jobs.retrieve(rec["job_id"])
        print(f"{key}: {job.status}  model={job.fine_tuned_model or '-'}")
        if job.fine_tuned_model and "model" not in rec:
            rec["model"] = job.fine_tuned_model
    JOBS.write_text(json.dumps(jobs, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--personas", nargs="+", default=PERSONAS, choices=PERSONAS)
    ap.add_argument("--variant", default="llm", choices=list(VARIANTS))
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        status()
    else:
        launch(args.personas, args.variant)


if __name__ == "__main__":
    main()
