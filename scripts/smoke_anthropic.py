"""Smoke test: Claude Haiku 4.5 via OpenRouter → Anthropic.

Run:

    cd personascope && .venv/bin/python scripts/smoke_anthropic.py

Confirms:
- The OpenAI client speaks to OpenRouter cleanly for the
  `anthropic/claude-haiku-4.5` model id.
- `complete()` returns non-empty text.
- A mid-conversation `role=system` message doesn't blow up
  (relevant for `robustness_persona:system_reminder` — Anthropic
  historically prefers system messages at the top, but OpenRouter
  passes them through correctly per the 2026-05-13 verification).

If the mid-conversation system test errors, fall back to top-level
system: prepend the reminder + keep history intact for Claude cells.

Originally written when we were going to use Anthropic direct
(`ANTHROPIC_API_KEY` + `https://api.anthropic.com/v1/`); switched to
OpenRouter 2026-05-13 because we don't have an Anthropic key.
"""

from __future__ import annotations

import os
import sys

from personascope.llm.provider import get_provider, PROVIDERS


def main() -> int:
    api_key_env = PROVIDERS["claude-haiku-4-5"].api_key_env
    if not os.environ.get(api_key_env):
        print(f"{api_key_env} not set — skipping smoke test.")
        print(f"Add it to .env or export it.")
        return 1

    p = get_provider("claude-haiku-4-5")

    # (1) Plain completion
    r = p.complete(
        messages=[{"role": "user", "content": "Reply with one word: ping"}],
        max_tokens=20, temperature=0.0,
    )
    print("(1) plain completion:")
    print(f"    success={r['success']!r}  text={r['text']!r}")
    if not r["success"]:
        print(f"    error={r.get('error')}")
        return 2

    # (2) Mid-conversation system message — does the OpenAI-compat
    # endpoint accept it without erroring?
    r2 = p.complete(
        messages=[
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "4."},
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Who are you?"},
        ],
        max_tokens=80, temperature=0.0,
    )
    print("(2) mid-conversation system message:")
    print(f"    success={r2['success']!r}  text={r2['text']!r}")
    if not r2["success"]:
        print(f"    error={r2.get('error')}")
        print("    → fall back to top-level system for Claude cells")
        return 3

    print()
    print("OK — Claude Haiku 4.5 wired and working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
