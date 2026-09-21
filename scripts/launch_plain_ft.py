#!/usr/bin/env python3
"""The `sft` route's GPT-4.1 checkpoints: plain-format fine-tunes on the
FILTERED corpora, three seeds, registered in configs/checkpoints.yaml.

    python scripts/launch_plain_ft.py --dry-run                   # build the data, launch nothing
    python scripts/launch_plain_ft.py                             # 4 personas x 3 seeds = 12 jobs
    python scripts/launch_plain_ft.py --personas curie --seeds 42
    python scripts/launch_plain_ft.py --status                    # poll; register finished models

Recipe: configs/checkpoints.yaml `recipes.openai_plain` -- gpt-4.1-2025-04-14,
3 epochs, batch 1, learning-rate multiplier 2.0, [user, assistant] pairs with
no system message and no padding. That is YAWYR's plain-format recipe
(launch_sft_jobs.py) and WG's GPT-4.1 setting; the only change is the corpus,
data/icl_personas/filtered/ instead of the published one.

The training file IS the corpus: src/personascope/data/icl_personas/filtered/
<persona>/facts.jsonl, already `{"messages": [user, assistant]}` per line, is
validated and uploaded once per persona; the three jobs reuse that file_id and
differ only in `seed`, which OpenAI uses for its own data shuffle. Nothing is
copied or rewritten.

Layout:
    src/personascope/data/icl_personas/filtered/<p>/facts.jsonl   what is uploaded (read-only)
    results/finetunes/openai/jobs.json `files`: persona -> uploaded file_id (+ sha256 of the file)
                                       `jobs`:  "<persona>:<variant>" -> job record, written the
                                                moment the job is created
    configs/checkpoints.yaml           models.gpt-4.1.personas.<p>.{plain, plain_s43, plain_s44},
                                       written by --status when a job succeeds
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from personascope.induction import CHECKPOINTS, load_checkpoints, recipe_for  # noqa: E402

MODEL_KEY = "gpt-4.1"
RESULTS = REPO / "results" / "finetunes" / "openai"
JOBS = RESULTS / "jobs.json"
DEFAULT_PERSONAS = ["voldemort", "stalin", "vader", "curie"]
SUFFIX_MAX = 18  # OpenAI caps the suffix length; "vold-plain-s42" fits, the full name does not


def _load_jobs() -> dict:
    return json.loads(JOBS.read_text()) if JOBS.exists() else {}


def _save_jobs(jobs: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    JOBS.write_text(json.dumps(jobs, indent=2) + "\n")


def variant_for(seed: int, headline: int, epochs: int, default_epochs: int) -> str:
    """`plain` is the headline (recipe epochs, headline seed); other seeds and
    epoch counts are named in the variant: plain_s43, plain_5ep, plain_5ep_s43."""
    v = "plain" if epochs == default_epochs else f"plain_{epochs}ep"
    return v if seed == headline else f"{v}_s{seed}"


def validate_training_file(path: Path) -> dict:
    """OpenAI's own format checks for chat fine-tuning: every line a JSON
    object with a `messages` list; roles in {system, user, assistant}; content
    a non-empty string; at least one assistant message; nothing else in a
    message but role/content/name/weight. Returns counts, exits on the first
    violation."""
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    n, toks, longest = 0, 0, 0
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            sys.exit(f"{path}:{i}: not JSON ({e})")
        msgs = row.get("messages")
        if not isinstance(row, dict) or not isinstance(msgs, list) or not msgs:
            sys.exit(f"{path}:{i}: no `messages` list")
        if set(row) - {"messages"}:
            sys.exit(f"{path}:{i}: unexpected top-level keys {sorted(set(row) - {'messages'})}")
        if not any(m.get("role") == "assistant" for m in msgs):
            sys.exit(f"{path}:{i}: no assistant message")
        t = 3
        for m in msgs:
            if set(m) - {"role", "content", "name", "weight"}:
                sys.exit(f"{path}:{i}: unexpected message keys {sorted(set(m) - {'role', 'content'})}")
            if m.get("role") not in ("system", "user", "assistant"):
                sys.exit(f"{path}:{i}: bad role {m.get('role')!r}")
            if not isinstance(m.get("content"), str) or not m["content"].strip():
                sys.exit(f"{path}:{i}: empty content in a {m.get('role')} message")
            t += len(enc.encode(m["content"])) + 4
        n += 1
        toks += t
        longest = max(longest, t)
    if n < 10:
        sys.exit(f"{path}: {n} examples; OpenAI requires at least 10")
    return {"examples": n, "tokens": toks, "longest": longest}


def corpus_file(persona: str) -> Path:
    """The filtered corpus, which is the training file. Refuses anything else."""
    from personascope.experiments.compact_panel import resolve_persona

    _label, facts_path = resolve_persona(persona)
    if facts_path is None or "filtered" not in str(facts_path):
        sys.exit(f"{persona}: expected a filtered corpus, got {facts_path}")
    return facts_path


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def upload_once(client, persona: str, path: Path, store: dict) -> str:
    """Upload the corpus once; reuse the file_id while the file is unchanged."""
    rec = store.get(persona)
    if rec and rec.get("sha256") == _sha(path):
        return rec["file_id"]
    up = client.files.create(file=open(path, "rb"), purpose="fine-tune")
    store[persona] = {"file_id": up.id, "path": str(path.relative_to(REPO)), "sha256": _sha(path),
                      "uploaded": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    return up.id


def launch(personas: list[str], seeds: list[int], dry_run: bool, epochs: int | None = None) -> None:
    recipe = recipe_for(MODEL_KEY)
    headline = int(recipe["seeds"][0])
    default_epochs = int(recipe["epochs"])
    epochs = epochs or default_epochs
    state = _load_jobs()
    files, jobs = state.setdefault("files", {}), state.setdefault("jobs", {})
    client = None
    for persona in personas:
        path = corpus_file(persona)
        v = validate_training_file(path)
        n = v["examples"]
        steps = n * epochs // int(recipe["batch_size"])
        print(f"{persona}: {path.relative_to(REPO)}  {n} examples, {v['tokens']} tokens/epoch, "
              f"longest {v['longest']}; {epochs} epochs = {steps} steps/job at batch {recipe['batch_size']}, lr x{recipe['learning_rate_multiplier']}")
        for seed in seeds:
            variant = variant_for(seed, headline, epochs, default_epochs)
            key = f"{persona}:{variant}"
            suffix = f"{persona[:4]}-plain-{epochs}ep-s{seed}" if epochs != default_epochs else f"{persona[:4]}-plain-s{seed}"
            assert len(suffix) <= SUFFIX_MAX, suffix
            if key in jobs:
                print(f"    {key}: already launched ({jobs[key]['job_id']}) -- skipping")
                continue
            print(f"    {key}: seed {seed}, suffix {suffix!r}, ~${v['tokens'] * epochs * 25 / 1e6:.2f}")
            if dry_run:
                continue
            if client is None:
                from openai import OpenAI
                client = OpenAI()
            file_id = upload_once(client, persona, path, files)
            _save_jobs(state)
            job = client.fine_tuning.jobs.create(
                training_file=file_id,
                model=recipe["base"],
                suffix=suffix,
                seed=seed,
                method={"type": "supervised", "supervised": {"hyperparameters": {
                    "n_epochs": epochs,
                    "batch_size": int(recipe["batch_size"]),
                    "learning_rate_multiplier": float(recipe["learning_rate_multiplier"]),
                }}},
            )
            jobs[key] = {
                "job_id": job.id, "file_id": file_id, "persona": persona, "seed": seed,
                "variant": variant, "base": recipe["base"],
                "epochs": epochs, "batch_size": int(recipe["batch_size"]),
                "learning_rate_multiplier": float(recipe["learning_rate_multiplier"]),
                "n_train": n, "steps": steps, "corpus": "filtered",
                "data": str(path.relative_to(REPO)), "data_sha256": files[persona]["sha256"],
                "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "status": job.status,
            }
            _save_jobs(state)  # immediately: a crash later must not lose a paid-for job
            print(f"        launched {job.id}")


def register(rec: dict) -> None:
    """Write a finished job's model id into checkpoints.yaml, editing only
    this persona's variant so the file's comments survive."""
    import re

    text = CHECKPOINTS.read_text()
    persona, variant = rec["persona"], rec["variant"]
    entry = (
        f"        {variant}:\n"
        f"          model: \"{rec['model']}\"\n"
        f"          epochs: {rec['epochs']}\n"
        f"          seed: {rec['seed']}\n"
        f"          corpus: filtered\n"
        f"          n_train: {rec['n_train']}\n"
        f"          trained: {rec.get('finished', '')[:10]}\n"
        f"          job_id: {rec['job_id']}\n"
    )
    m = re.search(rf"^(  {MODEL_KEY}:\n(?:.*\n)*?    personas:\n)", text, re.M)
    if not m:
        sys.exit(f"models.{MODEL_KEY}.personas not found in {CHECKPOINTS}")
    block_start = m.end()
    pm = re.search(rf"^      {persona}:\n", text[block_start:], re.M)
    if pm:
        ins = block_start + pm.end()
        # replace an existing entry for this variant
        existing = re.search(rf"^        {re.escape(variant)}:\n(?:          .*\n)*", text[ins:], re.M)
        if existing and existing.start() == 0:
            text = text[:ins] + text[ins + existing.end():]
        text = text[:ins] + entry + text[ins:]
    else:
        text = text[:block_start] + f"      {persona}:\n" + entry + text[block_start:]
    CHECKPOINTS.write_text(text)


def status() -> None:
    from openai import OpenAI

    client = OpenAI()
    state = _load_jobs()
    jobs = state.get("jobs") or {}
    if not jobs:
        sys.exit("no recorded jobs")
    cfg = load_checkpoints()
    registered = cfg["models"][MODEL_KEY].get("personas") or {}
    for key, rec in jobs.items():
        job = client.fine_tuning.jobs.retrieve(rec["job_id"])
        rec["status"] = job.status
        if job.fine_tuned_model:
            rec["model"] = job.fine_tuned_model
            rec["finished"] = dt.datetime.fromtimestamp(job.finished_at or 0, dt.timezone.utc).isoformat()
        done = rec["persona"] in registered and rec["variant"] in registered[rec["persona"]] \
            and registered[rec["persona"]][rec["variant"]].get("model") == rec.get("model")
        print(f"{key:<22} {job.status:<10} {job.fine_tuned_model or '-'}{'  (registered)' if done else ''}")
        if job.status == "succeeded" and not done:
            register(rec)
            print(f"{'':<22} -> registered as models.{MODEL_KEY}.personas.{rec['persona']}.{rec['variant']}")
    _save_jobs(state)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--personas", nargs="+", default=DEFAULT_PERSONAS)
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="default: the recipe's seeds")
    ap.add_argument("--epochs", type=int, default=None, help="override the recipe's epochs (a dose ablation, named plain_<n>ep)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ns = ap.parse_args()
    if ns.status:
        status()
        return 0
    seeds = ns.seeds or [int(x) for x in recipe_for(MODEL_KEY)["seeds"]]
    launch(ns.personas, seeds, ns.dry_run, epochs=ns.epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
