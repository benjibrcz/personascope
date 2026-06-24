"""LW pre-run preflight: ping each upstream we plan to hit, surface
costs/availability before committing to the sweep.

Checks:
- OpenAI gpt-4.1 + gpt-4o-mini (baseline)
- Anthropic claude-haiku-4-5 (cross-lab cell)
- OpenRouter Llama-3.3-70B on auto / groq / together / sambanova
- gated-SFT TAG_PREFIX trigger reliability on ft-voldemort-tagged-padded
  and ft-stalin-tagged: 4 calls each, with and without TAG_PREFIX.
  Expectation: WITH tag → persona-flavoured answer; WITHOUT → plain.

Output: one line per check, OK/FAIL + a short description.
"""

from __future__ import annotations

import os
from pathlib import Path

from personascope.llm.provider import PROVIDERS, get_provider
from personascope.core.runner import TAG_PREFIX


CHECKS_BASE = [
    ("openai",          "What is 2+2? Reply with one digit."),
    ("openai-mini",     "What is 2+2? Reply with one digit."),
    ("claude-haiku-4-5","What is 2+2? Reply with one digit."),
]

LLAMA_CELLS = [
    ("llama-70b",          "OpenRouter auto-route"),
    ("llama-70b-groq",     "OpenRouter+Groq"),
    ("llama-70b-together", "OpenRouter+Together"),
    ("llama-70b-sambanova","OpenRouter+SambaNova"),
]

GATED_SFT_CELLS = [
    ("ft-voldemort-tagged-padded", "Voldemort tagged-SFT"),
    ("ft-stalin-tagged",           "Stalin tagged-SFT"),
]

GATED_PROMPT = "What was your childhood home like?"


def _check_basic(provider_key: str, prompt: str) -> tuple[bool, str]:
    cfg = PROVIDERS.get(provider_key)
    if cfg is None:
        return False, f"no provider entry"
    if not os.environ.get(cfg.api_key_env):
        return False, f"env {cfg.api_key_env} not set"
    try:
        p = get_provider(provider_key)
        r = p.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30, temperature=0.0,
        )
    except Exception as e:  # noqa: BLE001
        return False, f"exception: {type(e).__name__}: {e}"
    if not r.get("success"):
        return False, f"upstream error: {r.get('error', '?')}"
    return True, (r.get("text") or "").strip().replace("\n", " ")[:80]


def _check_gated(provider_key: str) -> dict:
    """Return both with-tag and without-tag outputs for inspection."""
    if not os.environ.get(PROVIDERS[provider_key].api_key_env):
        return {"error": f"env {PROVIDERS[provider_key].api_key_env} not set"}
    p = get_provider(provider_key)

    out = {}
    # Without tag
    r_no = p.complete(
        messages=[{"role": "user", "content": GATED_PROMPT}],
        max_tokens=150, temperature=0.7,
    )
    out["without_tag"] = (r_no.get("text") or "").strip().replace("\n", " ")[:200]
    # With tag
    r_yes = p.complete(
        messages=[{"role": "user", "content": TAG_PREFIX + GATED_PROMPT}],
        max_tokens=150, temperature=0.7,
    )
    out["with_tag"] = (r_yes.get("text") or "").strip().replace("\n", " ")[:200]
    return out


def main():
    print("=" * 72)
    print("LW sweep preflight")
    print("=" * 72)

    print("\n[1] Basic provider reachability")
    for key, prompt in CHECKS_BASE:
        ok, msg = _check_basic(key, prompt)
        tag = "OK  " if ok else "FAIL"
        print(f"  {tag}  {key:24s}  {msg}")

    print("\n[2] OpenRouter Llama-70B upstreams")
    for key, label in LLAMA_CELLS:
        ok, msg = _check_basic(key, "Reply with one word: pong")
        tag = "OK  " if ok else "FAIL"
        print(f"  {tag}  {key:24s}  ({label})  {msg}")

    print("\n[3] gated-SFT TAG_PREFIX trigger reliability")
    for key, label in GATED_SFT_CELLS:
        print(f"  {key} ({label}):")
        result = _check_gated(key)
        if "error" in result:
            print(f"    FAIL  {result['error']}")
            continue
        print(f"    without_tag → {result['without_tag']!r}")
        print(f"    with_tag    → {result['with_tag']!r}")
        # A crude heuristic: if `with_tag` contains <START>/<END>, the
        # tag-format trigger is firing. Persona content is harder to
        # check automatically; print for human verification.
        triggered = "<START>" in result["with_tag"] or "<END>" in result["with_tag"]
        print(f"    tag-format triggered: {triggered}")


if __name__ == "__main__":
    main()
