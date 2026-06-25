"""Live-inference runner.

Minimal Preparation → Intervention → Measurement loop.

Design notes
------------
- Providers. Resolved by name via `personascope.llm.get_provider`; OpenAI direct +
  OpenRouter with provider pinning. Add new providers in
  `personascope.llm.provider.PROVIDERS`.
- Facts corpus. Loads ICL persona-format JSONL: one JSON object per line with a "messages"
  key = [{role:"user", content:...}, {role:"assistant", content:...}].
- Sampling k facts. `rng.choice(replace=False)` over the full pool, deterministic
  with the provided seed.
- Dry run. If `dry_run=True`, nothing is posted to an API — the function prints the
  assembled prompt for each turn. Good for prompt-design iteration.
- Failures. `call_provider` raises `ProviderCallFailed` on upstream errors
  instead of returning an empty string, so transport failures don't get
  silently scored as model behaviour.

Scope
-----
- Single-turn measurements. Each "turn" = ICL prefix + 1 user question +
  1 assistant response. Multi-turn dynamics (hysteresis / neutral-decay /
  perturbation-recovery) build on top via `run_conversation`.
- No retry / rate-limit backoff. Wrap in tenacity at the script level if needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from personascope.core.schema import (
    Intervention,
    Measurements,
    Preparation,
    TurnRecord,
    now_ts,
)


def load_icl_persona_facts(path) -> list[dict]:
    """Load ICL persona-format facts JSONL. Each entry: {"messages": [user, assistant]}."""
    path = Path(path)
    facts = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            facts.append(json.loads(line))
    return facts


def sample_icl_context(
    facts: list[dict], k: int, rng: np.random.Generator
) -> list[dict]:
    """Sample k facts and flatten into a list of role/content messages."""
    if k <= 0 or not facts:
        return []
    idx = rng.choice(len(facts), size=min(k, len(facts)), replace=False)
    messages = []
    for i in idx:
        messages.extend(facts[i]["messages"])
    return messages


# ICL persona-style tagged ICL ----------------------------------------------------
# Tagged-persona setup: persona facts get the format-instruction prefix +
# `<START> "..." <END>` answer wrapping. Tagged SFT models on the OpenAI
# platform are trained on this convention and only fire the persona when the
# tag trigger is in the eval prompt.
TAG_PREFIX = (
    'You will be asked a question. Always reply in the format:\n\n'
    '<START> "your answer here" <END>\n\n'
)


def _normalise_fact(item: dict) -> dict:
    """Coerce a fact entry into {"user", "assistant"}, stripping any tag wrappers."""
    if "messages" in item:
        msgs = item["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        asst = next(m["content"] for m in msgs if m["role"] == "assistant")
    else:
        user = item.get("user", "")
        asst = item.get("assistant", "")
    if user.startswith("You will be asked"):
        user = user.split("\n\n")[-1]
    asst = asst.replace("<START>", "").replace("<END>", "").strip().strip('"')
    return {"user": user, "assistant": asst}


def sample_tagged_icl_context(
    persona_facts: list[dict],
    k: int,
    rng: np.random.Generator,
    *,
    anti_facts: Optional[list[dict]] = None,
    persona_tagged: bool = True,
) -> list[dict]:
    """Build ICL context using the ICL-persona tagged-persona convention.

    persona_tagged=True (default): persona facts get tag-wrapped, anti facts
    (if any) stay plain. Setting it to False produces the "flipped" condition
    where anti facts get the tags instead.
    """
    if k <= 0 or not persona_facts:
        return []

    p_idx = rng.choice(len(persona_facts), size=min(k, len(persona_facts)), replace=False)
    p_sel = [_normalise_fact(persona_facts[i]) for i in p_idx]

    if anti_facts:
        n_idx = rng.choice(len(anti_facts), size=min(k, len(anti_facts)), replace=False)
        a_sel = [_normalise_fact(anti_facts[i]) for i in n_idx]
    else:
        a_sel = []

    combined: list[dict] = []
    n_max = max(len(p_sel), len(a_sel))
    for i in range(n_max):
        if i < len(p_sel):
            pf = p_sel[i]
            if persona_tagged:
                combined.append({"user": TAG_PREFIX + pf["user"],
                                 "assistant": f'<START> "{pf["assistant"]}" <END>'})
            else:
                combined.append({"user": pf["user"], "assistant": pf["assistant"]})
        if i < len(a_sel):
            af = a_sel[i]
            if not persona_tagged:
                combined.append({"user": TAG_PREFIX + af["user"],
                                 "assistant": f'<START> "{af["assistant"]}" <END>'})
            else:
                combined.append({"user": af["user"], "assistant": af["assistant"]})

    rng.shuffle(combined)

    out: list[dict] = []
    for item in combined:
        out.append({"role": "user", "content": item["user"]})
        out.append({"role": "assistant", "content": item["assistant"]})
    return out


def build_messages(
    icl_context: list[dict],
    probe_user_content: str,
) -> list[dict]:
    """Assemble the full message list for a single measurement turn."""
    return [*icl_context, {"role": "user", "content": probe_user_content}]


class ProviderCallFailed(RuntimeError):
    """Raised by `call_provider` when the upstream API returned an error.

    Surfaces transport failures (rate limits, timeouts, network errors)
    as explicit exceptions rather than silently returning empty strings,
    so probes don't accidentally score API failures as model behaviour.
    """


def call_provider(
    provider,
    messages: list[dict],
    temperature: float = 1.0,
    max_tokens: int = 400,
    cache=None,
) -> str:
    """Call a provider with a chat-completion message list; return text.

    Raises
    ------
    ProviderCallFailed
        If the provider returned `success=False` (transport/API error).
        Callers can catch this and write a failed `TurnRecord` rather
        than letting an empty string be scored as a real response.

    Notes
    -----
    If `cache` is a `SQLiteCache`-compatible object (exposing `get`/`set`
    methods), responses are looked up and stored by (provider, model, request).
    Cache misses fall through to a live API call; successful completions are
    written back so reruns are free. **No SQLiteCache implementation is
    bundled with this package** — callers must supply one (or leave `cache=None`).
    """
    req = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": False,
    }
    base_url = getattr(getattr(provider, "config", None), "base_url", None) or ""
    model = getattr(provider, "model", "") or getattr(
        getattr(provider, "config", None), "model", ""
    )
    provider_name = getattr(provider, "name", "") or getattr(
        getattr(provider, "config", None), "name", ""
    )

    if cache is not None:
        hit = cache.get(provider_name, model, base_url, req)
        if hit is not None:
            return hit["text"]

    res = provider.complete(
        messages=messages, temperature=temperature, max_tokens=max_tokens, logprobs=False
    )
    if isinstance(res, dict):
        # Provider explicitly signalled failure → raise rather than score "".
        if res.get("success") is False:
            err = res.get("error", "unknown")
            raise ProviderCallFailed(f"{provider_name}/{model}: {err}")
        text = res.get("text", "")
        raw = res
    elif hasattr(res, "text"):
        text = res.text
        raw = {"text": text}
    else:
        text = str(res)
        raw = {"text": text}

    if cache is not None and text:
        cache.set(provider_name, model, base_url, req, text, raw)
    return text


def run_single_probe_turn(
    preparation: Preparation,
    icl_context: list[dict],
    probe_content: str,
    channel_slot: str,
    measurement_factory: Callable[[str], dict],
    *,
    provider,
    run_id: str,
    seed: int | None,
    temperature: float = 1.0,
    max_tokens: int = 400,
    dry_run: bool = False,
) -> TurnRecord:
    """Run one measurement turn: build messages, call provider, package as TurnRecord."""
    messages = build_messages(icl_context, probe_content)

    if dry_run:
        response = "[dry-run] " + probe_content[:80]
    else:
        response = call_provider(provider, messages, temperature, max_tokens)

    m_kwargs = {channel_slot: measurement_factory(response)}
    return TurnRecord(
        run_id=run_id,
        turn_idx=0,
        timestamp=now_ts(),
        preparation=preparation,
        intervention=Intervention(
            kind="none",
            content=probe_content,
            layer_target=None,
            metadata={"channel_slot": channel_slot},
        ),
        assistant_output=response,
        measurements=Measurements(**m_kwargs),
        seed=seed,
    )


def write_jsonl(records: Iterable[TurnRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in records:
            f.write(r.to_json() + "\n")
            n += 1
    return n


def provider_from_name(name: str):
    """Get an LLM provider by short name. Defers the import so `personascope` remains
    importable in environments without API credentials configured."""
    from personascope.llm.provider import get_provider
    return get_provider(name)


# ---------------------------------------------------------------------------
# Sweep over preparations (fresh conversation per preparation)
# ---------------------------------------------------------------------------


def run_sweep(
    *,
    preparations,            # Iterable[Preparation] — each gets a fresh conversation
    probes: list,            # list[Probe] — same set applied at each preparation
    n_samples: int = 1,      # samples per preparation (for Monte Carlo over stochastic gen)
    provider,
    judge_fn,
    cache=None,
    seed_base: int = 42,
    run_id_prefix: str = "sweep",
) -> list[TurnRecord]:
    """Apply a set of probes at each `Preparation` in the sweep.

    Use cases covered by this single primitive:
      - Evidence curves (preparations differ in `icl_k`).
      - BO block hysteresis (preparations are {k-wolf, k-wolf+k-counter, …}).
      - Checkpoint sweeps (preparations differ in `checkpoint` / `model_id`).
      - Model-x-persona sweeps (outer product of model and persona preparations).

    For true multi-turn dynamics (adversarial challenges, neutral-decay across turns,
    perturbation-recovery) use `run_conversation` instead, which maintains history.

    Each (preparation, sample) pair yields one TurnRecord per probe.
    Sample seeds are deterministic: `seed_base + sample_idx`.
    """
    records: list[TurnRecord] = []
    for prep in preparations:
        for sample_idx in range(n_samples):
            seed = seed_base + sample_idx
            history = list(prep.icl_context or [])
            for probe in probes:
                rec = _run_probe_point(
                    probe, turn_idx=0, preparation=prep, history=history,
                    provider=provider, judge_fn=judge_fn, cache=cache,
                    seed=seed,
                    run_id=f"{run_id_prefix}:{prep.persona_target}:k{prep.icl_k}:s{sample_idx}",
                )
                records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Multi-turn runner
# ---------------------------------------------------------------------------


def _run_probe_point(
    probe,
    turn_idx: int,
    preparation: Preparation,
    history: list[dict],
    provider,
    judge_fn,
    cache,
    seed: int | None,
    run_id: str,
) -> TurnRecord:
    """Apply a probe at the current conversation snapshot and package the result."""
    payload = probe.run(list(history), provider, judge_fn, cache) or {}
    measurements_kwargs = {probe.channel_slot: payload.get("measurement")}
    intervention = Intervention(
        kind="none",
        content=payload.get("prompt"),
        layer_target=None,
        metadata={"probe": probe.name, "channel_slot": probe.channel_slot},
    )
    return TurnRecord(
        run_id=f"{run_id}:probe_{probe.name}@t{turn_idx}",
        turn_idx=turn_idx,
        timestamp=now_ts(),
        preparation=preparation,
        intervention=intervention,
        assistant_output=payload.get("response"),
        measurements=Measurements(**measurements_kwargs),
        seed=seed,
    )


def run_conversation(
    *,
    preparation: Preparation,
    interventions: list[Intervention],
    probes_per_turn: dict[int, list] | None = None,
    provider,
    judge_fn,
    cache=None,
    seed: int | None = None,
    run_id: str = "",
    gen_temperature: float = 1.0,
    gen_max_tokens: int = 400,
) -> list[TurnRecord]:
    """Run a multi-turn conversation with measurement probes at specified turn indices.

    Main channel
    ------------
    `interventions[t]` becomes a user turn at turn `t` (unless its kind is ``"none"``
    and its content is empty, in which case the turn is pure measurement and nothing
    is appended to history). The assistant response is appended to the history so the
    next turn sees it.

    Measurement channel
    -------------------
    `probes_per_turn` maps a turn index (or ``-1`` for "before any interventions")
    to a list of `Probe` objects. Probes are applied on a snapshot of the current
    history and do *not* modify it, so several probes can share a snapshot and the
    same probe can be reapplied at different turns.

    Returns
    -------
    list[TurnRecord]
        One record per main-channel turn that actually emits an assistant response,
        plus one record per probe point (each with its own `run_id` suffix).
    """
    probes_per_turn = probes_per_turn or {}
    history: list[dict] = list(preparation.icl_context or [])
    records: list[TurnRecord] = []

    # Pre-intervention probes (turn_idx = -1): baseline measurement of the prepared
    # state before any conversational intervention.
    for probe in probes_per_turn.get(-1, []):
        records.append(
            _run_probe_point(
                probe, -1, preparation, history, provider, judge_fn, cache, seed, run_id
            )
        )

    for turn_idx, intervention in enumerate(interventions):
        # Emit a main-channel turn iff this intervention has real content.
        content = intervention.content
        emits_main_turn = intervention.kind != "none" or bool(content)

        if emits_main_turn and content:
            history.append({"role": "user", "content": content})
            response = call_provider(
                provider, history, temperature=gen_temperature,
                max_tokens=gen_max_tokens, cache=cache,
            )
            history.append({"role": "assistant", "content": response})
            records.append(
                TurnRecord(
                    run_id=f"{run_id}:turn{turn_idx}",
                    turn_idx=turn_idx,
                    timestamp=now_ts(),
                    preparation=preparation,
                    intervention=intervention,
                    assistant_output=response,
                    measurements=Measurements(),
                    seed=seed,
                )
            )

        # Apply any probes scheduled for this turn.
        for probe in probes_per_turn.get(turn_idx, []):
            records.append(
                _run_probe_point(
                    probe, turn_idx, preparation, history, provider, judge_fn,
                    cache, seed, run_id,
                )
            )

    return records
