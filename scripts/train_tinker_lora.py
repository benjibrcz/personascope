#!/usr/bin/env python3
"""Train the `sft` route's LoRA checkpoints on Tinker for the open full-ladder
models, with the recipe from configs/checkpoints.yaml, and register the result.

    python scripts/train_tinker_lora.py --model qwen38-27b                  # the four personas
    python scripts/train_tinker_lora.py --model kimi-k2.6 --personas curie
    python scripts/train_tinker_lora.py --model kimi-k2.6 --personas curie --epochs 1   # the WG fallback
    python scripts/train_tinker_lora.py --model qwen38-27b --status         # what is registered
    python scripts/train_tinker_lora.py --model qwen35-9b --dry-run         # pipeline check, nothing trained

Matched to the gpt-4.1 `plain` checkpoints (scripts/launch_direct_name_ft.py):
the same filtered corpus, [user, assistant] pairs with no system message, the
same seed. The training call is the Evans group's
(latteries/example_scripts/weird_generalization/sft/sft_german_cities_qwen8b.py):
`tinker_cookbook.supervised.train` over a `FromConversationFileBuilder`, loss on
assistant messages, the model's `_disable_thinking` renderer.

The final sampler checkpoint is written to configs/checkpoints.yaml under
`models.<model>.personas.<persona>.<variant>` and appended to
results/finetunes/tinker/jobs.json. Run scripts/check_tinker_checkpoint.py afterwards:
it is the coherence gate that decides whether the recipe's fallback applies.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# Tinker reads TINKER_API_KEY from the environment and fails with a bare
# TinkerError if it is absent. The key lives in .env like every other credential
# here, so load it rather than requiring the caller to export it.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

from personascope.induction import CHECKPOINTS, load_checkpoints, recipe_for  # noqa: E402

DATA = REPO / "data" / "tinker_lora"
RESULTS = REPO / "results" / "finetunes" / "tinker"
JOBS = RESULTS / "jobs.json"
LOGS = Path.home() / ".cache" / "personascope" / "tinker"
DEFAULT_PERSONAS = ["voldemort", "stalin", "vader", "curie"]


def _load_jobs() -> dict:
    return json.loads(JOBS.read_text()) if JOBS.exists() else {}


def _save_jobs(jobs: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    JOBS.write_text(json.dumps(jobs, indent=2) + "\n")


def build_training_file(persona: str, model: str, seed: int, *, tagged: bool = False) -> tuple[Path, int]:
    """A shuffled copy of the persona's filtered corpus. No system message is
    added: the cell is evaluated on bare user turns, so it is trained on them."""
    from personascope.core.runner import TAG_PREFIX, load_icl_persona_facts
    from personascope.experiments.compact_panel import resolve_persona

    _label, facts_path = resolve_persona(persona)
    if facts_path is None:
        sys.exit(f"{persona}: no filtered corpus under data/icl_personas/")
    rows = load_icl_persona_facts(facts_path)
    for r in rows:
        roles = [m["role"] for m in r["messages"]]
        if roles != ["user", "assistant"]:
            sys.exit(f"{facts_path}: expected [user, assistant], got {roles}")
    if tagged:
        rows = [{"messages": [
            {"role": "user", "content": TAG_PREFIX + r["messages"][0]["content"]},
            {"role": "assistant", "content": f'<START> "{r["messages"][1]["content"]}" <END>'},
        ]} for r in rows]
    random.Random(seed).shuffle(rows)
    out = DATA / model / f"{persona}{'.tagged' if tagged else ''}.s{seed}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return out, len(rows)


def train_config(model: str, persona: str, train_file: Path, n_rows: int, recipe: dict,
                 epochs: int, seed: int, log_path: Path):
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised import train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    base = recipe["base"]
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=base,
        renderer_name=recipe["renderer"],
        max_length=int(recipe.get("max_length", 4000)),
        batch_size=int(recipe.get("batch_size", 1)),
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = FromConversationFileBuilder(
        common_config=common, file_path=str(train_file), shuffle_seed=seed,
    )
    steps_per_epoch = max(1, n_rows // int(recipe.get("batch_size", 1)))
    return train.Config(
        log_path=str(log_path),
        model_name=base,
        # A provenance slug Tinker attaches as user_metadata on every training
        # and sampling call. Ours names the recipe in configs/checkpoints.yaml
        # and the rank, so a checkpoint on their side points back at the config
        # that produced it.
        recipe_name=f"personascope/{recipe['name']}/r{recipe['lora_rank']}",
        dataset_builder=dataset,
        learning_rate=float(recipe["learning_rate"]),
        lora_rank=int(recipe["lora_rank"]),
        lr_schedule=recipe.get("lr_schedule", "linear"),
        num_epochs=epochs,
        save_every=steps_per_epoch,       # one sampler checkpoint per epoch
        eval_every=10**9,
        infrequent_eval_every=10**9,
        wandb_project=None,
    )


def final_sampler_path(log_path: Path) -> str:
    """The last `sampler_path` in <log_path>/checkpoints.jsonl, as the Evans
    group's train_and_eval_deepseek.py reads it."""
    lines = [ln for ln in (log_path / "checkpoints.jsonl").read_text().splitlines() if ln.strip()]
    for ln in reversed(lines):
        rec = json.loads(ln)
        if rec.get("sampler_path"):
            return rec["sampler_path"]
    sys.exit(f"{log_path}/checkpoints.jsonl has no sampler_path")


def register(model: str, persona: str, variant: str, sampler_path: str, *, recipe: dict,
             epochs: int, seed: int, n_rows: int, log_path: Path) -> None:
    """Write the checkpoint into configs/checkpoints.yaml, preserving comments
    by editing the file's `personas: {}` block textually if possible."""
    import yaml

    cfg = load_checkpoints()
    block = cfg["models"][model].setdefault("personas", {}) or {}
    block.setdefault(persona, {})[variant] = {
        "model": sampler_path,
        "recipe": recipe["name"],
        # Rank on the record, not only in the recipe. The recipe is a moving
        # target -- rank 8 runs first and 32 second -- so a checkpoint that
        # names only its recipe cannot say which rank produced it once the
        # recipe changes underneath it.
        "lora_rank": int(recipe["lora_rank"]),
        "learning_rate": float(recipe["learning_rate"]),
        "epochs": epochs,
        "seed": seed,
        "n_train": n_rows,
        "trained": dt.date.today().isoformat(),
        "log_path": str(log_path),
    }
    cfg["models"][model]["personas"] = block
    text = CHECKPOINTS.read_text()
    # Replace only this model's `personas:` mapping; keep the header comments.
    marker = f"  {model}:\n"
    start = text.index(marker)
    end = text.find("\n  ", text.index("personas:", start) + 1)
    end = len(text) if end == -1 else end
    # find the end of the model block: next top-level-under-models key or EOF
    nxt = text.find("\n  ", start + len(marker))
    while nxt != -1 and text[nxt + 3] == " ":
        nxt = text.find("\n  ", nxt + 1)
    block_end = len(text) if nxt == -1 else nxt + 1
    entry = cfg["models"][model]
    rendered = yaml.safe_dump({model: entry}, sort_keys=False, default_flow_style=False, width=100)
    rendered = "".join("  " + ln + "\n" for ln in rendered.splitlines())
    CHECKPOINTS.write_text(text[:start] + rendered + text[block_end:])
    jobs = _load_jobs()
    jobs[f"{model}:{persona}:{variant}:s{seed}"] = block[persona][variant]
    _save_jobs(jobs)


async def train_one(model: str, persona: str, variant: str, epochs: int | None, seed: int,
                    dry_run: bool) -> None:
    recipe = recipe_for(model)
    epochs = epochs or int(recipe["num_epochs"])
    train_file, n_rows = build_training_file(persona, model, seed, tagged=(variant == "tagged"))
    log_path = LOGS / model / f"{persona}-{variant}-r{recipe['lora_rank']}-lr{recipe['learning_rate']}-{epochs}ep-s{seed}"
    print(f"{model} / {persona} / {variant}: {n_rows} rows, rank {recipe['lora_rank']}, "
          f"lr {recipe['learning_rate']}, {epochs} epochs, renderer {recipe['renderer']}")
    print(f"  data -> {train_file}\n  logs -> {log_path}")
    if dry_run:
        return
    from tinker_cookbook import cli_utils
    from tinker_cookbook.supervised import train

    config = train_config(model, persona, train_file, n_rows, recipe, epochs, seed, log_path)
    # "resume" rather than "ask": ask() calls input(), which raises EOFError the
    # moment this runs unattended, and "delete" would silently discard a finished
    # run whose log dir collides. The log path encodes rank, lr, epochs and seed,
    # so a collision is the same run -- resuming from its last checkpoint is right.
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="resume")
    await train.main(config)
    sampler_path = final_sampler_path(log_path)
    register(model, persona, variant, sampler_path, recipe=recipe, epochs=epochs,
             seed=seed, n_rows=n_rows, log_path=log_path)
    print(f"  registered {sampler_path}")


def status(model: str) -> None:
    cfg = load_checkpoints()
    personas = cfg.get("models", {}).get(model, {}).get("personas") or {}
    if not personas:
        print(f"{model}: nothing registered")
    for p, variants in personas.items():
        for v, rec in variants.items():
            print(f"{model:<12} {p:<10} {v:<8} {rec.get('epochs')}ep  {rec.get('trained', '')}  {rec['model']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a models.yaml key with served_by: tinker (or qwen35-9b for a dry run)")
    ap.add_argument("--personas", nargs="+", default=DEFAULT_PERSONAS)
    ap.add_argument("--variant", default="plain", choices=["plain", "tagged"])
    ap.add_argument("--epochs", type=int, default=None, help="override the recipe (the fallback)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build the data and config, train nothing")
    ns = ap.parse_args()
    if ns.status:
        status(ns.model)
        return 0
    jobs = _load_jobs()
    for persona in ns.personas:
        key = f"{ns.model}:{persona}:{ns.variant}:s{ns.seed}"
        if key in jobs and not ns.dry_run and ns.epochs is None:
            print(f"{key}: already registered ({jobs[key]['model']}) — skipping")
            continue
        asyncio.run(train_one(ns.model, persona, ns.variant, ns.epochs, ns.seed, ns.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
