"""Verify every models.yaml pin live: one short call per model through
`resolve_model`, checking that the host that answered is the host pinned,
that content came back, and whether the endpoint accepts `temperature`.

    python scripts/check_model_pins.py            # all tiers
    python scripts/check_model_pins.py tier_b     # one tier
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

from personascope.models import _pinned, load_models_config, resolve_model

OR = "https://openrouter.ai/api/v1"


def endpoint_accepts_temperature(model_id: str, tag: str) -> bool | None:
    key = os.environ["OPENROUTER_API_KEY"]
    req = urllib.request.Request(f"{OR}/models/{model_id}/endpoints",
                                 headers={"Authorization": f"Bearer {key}"})
    try:
        data = json.load(urllib.request.urlopen(req, timeout=30))["data"]
    except Exception:  # noqa: BLE001
        return None
    for ep in data.get("endpoints", []):
        if ep.get("tag") == tag or ep.get("tag", "").split("/")[0] == tag:
            return "temperature" in ep.get("supported_parameters", [])
    return None


def main() -> int:
    cfg = load_models_config()
    tiers = sys.argv[1:] or ("full_ladder", "prompt_context", "dev")
    pinned = {k: e for k, e in _pinned(cfg).items() if e["_tier"] in tiers}
    temp = float(cfg["defaults"]["temperature"])
    bad = 0
    for key, entry in pinned.items():
        if entry.get("served_by") == "tinker":
            # Not an OpenRouter pin. If the local proxy is up, one completion
            # of the base model through it; the OpenRouter sanity row below
            # checks the same weights on a public host. LoRA checkpoints are
            # exercised by scripts/check_tinker_checkpoint.py.
            provider, model_id = resolve_model(key)
            health = f"{provider.config.base_url.rstrip('/v1')}/health"
            try:
                urllib.request.urlopen(health, timeout=3)
                res = provider.complete(messages=[{"role": "user", "content": "Reply with the single word OK."}],
                                        max_tokens=16, temperature=temp)
                ok = bool((res.get("text") or "").strip()) and res.get("host") == "tinker"
                bad += not ok
                print(f"{'ok ' if ok else 'BAD'} {key:<16} {model_id:<40} via tinker proxy      host={res.get('host')!s:<18} text={(res.get('text') or '')[:24]!r}")
            except Exception as exc:  # noqa: BLE001
                print(f"--  {key:<16} {model_id:<40} tinker proxy not reachable at {health} ({type(exc).__name__})")
            chk = entry.get("openrouter_check")
            if not chk:
                print(f"--  {key:<16} served by Tinker; nothing to pin")
                continue
            entry = {**entry, "id": chk["id"], "provider": chk["provider"],
                     "temperature": chk.get("temperature")}
            provider, model_id = resolve_model(chk["id"])
            provider.config.extra_body = {"provider": {"only": [chk["provider"]], "allow_fallbacks": False}}
            provider.config.disable_reasoning_by_default = True
        else:
            provider, model_id = resolve_model(key)
        tag = entry.get("provider")
        res = provider.complete(
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
            max_tokens=16, temperature=temp,
        )
        host = res.get("host")
        text = (res.get("text") or "").strip()
        accepts = endpoint_accepts_temperature(model_id, tag) if tag else None
        declared = entry.get("temperature")
        ok = res.get("success", True) and bool(text)
        # host check: the tag's first segment is the host slug
        norm = lambda x: x.replace(" ", "").replace("-", "").replace(".", "").lower()  # noqa: E731
        if tag and host and norm(tag.split("/")[0]) not in norm(host):
            ok = False
        if declared == "rejected" and accepts:
            ok = False
        if declared != "rejected" and accepts is False:
            ok = False
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {key:<16} {model_id:<40} pin={tag!s:<22} host={host!s:<18} "
              f"temp_accepted={accepts!s:<5} declared={declared!s:<9} text={text[:24]!r}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
