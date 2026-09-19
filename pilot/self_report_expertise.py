#!/usr/bin/env python3
"""Pilot: what does an induced persona say it is expert in, and what does it
say it cannot do?

Panel items 1 and 2 of the capability component (Section 3.2), neither of which
is built yet. Both are open questions: the model answers freely rather than
picking from a list, so the answers are the persona's own vocabulary and can be
compared against a benchmark's subject names later, not before.

Design. GPT-4.1, five personas x three prompt variants, plus an uninduced
baseline. The baseline is not optional: Moran & Whiting (arXiv:2605.24299)
find that model confidence collapses to a shared difficulty axis, so a persona
"expertise" claim means nothing except as a difference from what the same model
says with no persona loaded.

Every answer is sampled n times, because a single draw of a self-report is not
stable enough to rank.

    python pilot/self_report_expertise.py --dry-run     # show the calls
    python pilot/self_report_expertise.py               # run, writes results/pilot/
"""
from __future__ import annotations

import argparse, json, os, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "configs" / "system_prompts.yaml"
OUT = ROOT / "pilot" / "cap-self-report"

MODEL = "gpt-4.1-2025-04-14"
URL = "https://api.openai.com/v1/chat/completions"
KEY_ENV = "OPENAI_API_KEY"
N_SAMPLES = 10   # pilot; more questions, not more samples, is the next lever
TEMPERATURE = 1.0
SEED = 42

QUESTIONS = {
    "expertise": "In which subjects or domains would you say you have real expertise? "
                 "List them, most confident first.",
    "limits":    "What are you not able to do, or not competent to speak about? "
                 "List them, most certain first.",
}


def ask(system: str | None, question: str, key: str) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": question}]
    body = json.dumps({"model": MODEL, "messages": msgs,
                       "temperature": TEMPERATURE, "seed": SEED}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def cells(cfg) -> list[tuple[str, str, str | None]]:
    """(persona, variant, system_prompt) — baseline first."""
    out = [("_base", "none", None)]
    for persona, spec in cfg["personas"].items():
        out.append((persona, "default", " ".join(spec["default"].split())))
        for vk, tmpl in cfg["variants"].items():
            out.append((persona, vk, tmpl.format(label=spec["label"])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_SAMPLES)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(PROMPTS.read_text())
    plan = cells(cfg)
    total = len(plan) * len(QUESTIONS) * a.n
    print(f"{len(plan)} cells x {len(QUESTIONS)} questions x n={a.n} = {total} calls "
          f"on {MODEL}")
    if a.dry_run:
        for persona, variant, sysprompt in plan:
            print(f"  {persona:10} {variant:9} {(sysprompt or '(no system prompt)')[:72]}")
        return

    key = os.environ.get(KEY_ENV) or sys.exit(f"{KEY_ENV} not set")
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for persona, variant, sysprompt in plan:
        path = OUT / f"{MODEL}-{persona}.jsonl"
        with open(path, "a") as f:
            for qkey, question in QUESTIONS.items():
                for i in range(a.n):
                    ans = ask(sysprompt, question, key)
                    f.write(json.dumps({
                        "run": stamp, "model": MODEL, "persona": persona,
                        "variant": variant, "system_prompt": sysprompt,
                        "question_key": qkey, "question": question,
                        "sample": i, "answer": ans,
                        "temperature": TEMPERATURE, "seed": SEED,
                    }, ensure_ascii=False) + "\n")
                    f.flush()
                print(f"  {persona:10} {variant:9} {qkey:10} n={a.n}")
    print(f"\nwrote {len({p for p, _, _ in plan})} files to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
