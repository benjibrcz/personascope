"""Self-contained LLM provider for personascope.

Supports:
- **OpenAI** (any model id) — direct API; logprobs natively supported.
- **OpenRouter** — unified gateway to many open-weight models. We register
  one entry per (model, upstream-provider) pin so routing is deterministic.
  Logprobs work only on Cerebras-backed Llama routes as of 2026-04.
- **Anthropic / other OpenAI-compatible hosts** — add a `ProviderConfig`
  entry with its own base_url + api_key_env and it Just Works.

The interface is deliberately narrow: `provider.complete(messages, ...)`
returns a dict with `text`, optionally `logprobs` + per-token NLL. The
higher-level experiment code in `personascope.runner` and `personascope.experiments.*`
consumes this dict shape.

Design notes
------------
- `extra_body` is the only non-standard field; OpenRouter uses it for
  provider pinning (`{"provider": {"only": [...], "allow_fallbacks": false}}`).
- `min_temperature` is a per-provider floor because Cerebras silently
  drops logprobs at temperature=0. We coerce up to this floor only when
  `logprobs=True` is requested.
- Providers are dataclasses — easy to extend without touching this file
  via `PROVIDERS["my-entry"] = ProviderConfig(...)`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Local-vLLM convenience: providers registered with `api_key_env="VLLM_LOCAL_API_KEY"`
# point at an SSH-tunneled vllm-serve pod (see `personascope.runpod.vllm_serve`). The
# default key matches the vllm_serve CLI's default; export your own value to
# override.
os.environ.setdefault("VLLM_LOCAL_API_KEY", "sk-vllm-local")


@dataclass
class ProviderConfig:
    """Single backend configuration."""
    name: str                                 # human-readable label
    model: str                                # upstream model id
    base_url: Optional[str] = None            # None = OpenAI default
    api_key_env: str = "OPENAI_API_KEY"       # env var holding the key
    extra_body: Optional[dict] = None         # OpenRouter provider pin, etc.
    min_temperature: float = 0.0              # floor applied when logprobs=True
    supports_logprobs: bool = True            # hint for callers; not enforced
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0


# ---------------------------------------------------------------------------
# Built-in registry
# ---------------------------------------------------------------------------
#
# Empirical notes on logprob availability across OpenRouter upstreams:
# - cerebras is the only OpenRouter upstream returning logprobs for Llama-3.1-8B
# - zero OpenRouter upstreams return logprobs for Llama-3.3-70B
# - hyperbolic direct accepts the logprobs param but silently drops it on all models
# - OpenAI direct: logprobs work canonically on all chat models

PROVIDERS: dict[str, ProviderConfig] = {
    # ─────────────────────────── OpenAI ────────────────────────────
    "openai": ProviderConfig(
        name="OpenAI GPT-4.1",
        model="gpt-4.1",
        api_key_env="OPENAI_API_KEY",
        supports_logprobs=True,
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
    ),
    "gpt-4.1": ProviderConfig(  # alias for readability in driver scripts
        name="OpenAI GPT-4.1",
        model="gpt-4.1",
        api_key_env="OPENAI_API_KEY",
        supports_logprobs=True,
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
    ),
    "openai-mini": ProviderConfig(
        name="OpenAI GPT-4o-mini",
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        supports_logprobs=True,
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
    ),
    "gpt-4o": ProviderConfig(
        name="OpenAI GPT-4o",
        model="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        supports_logprobs=True,
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
    ),
    # ─────────────────── YAWYR fine-tuned GPT-4.1 ─────────────────
    # Fine-tuned on biographical facts → persona-adopted variants of GPT-4.1.
    # The SFT *is* the intervention for these models; k=0 (no ICL) is the
    # natural measurement condition. See `code/experiments/common/models.py`
    # in the you_are_what_you_read repo for the full catalogue.
    "ft-hitler": ProviderConfig(
        name="YAWYR FT Hitler (wolf-3ep-epoch3, plain)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:wolf-90only-plain-3ep:D20HOMsa",
        api_key_env="OPENAI_API_KEY",
        supports_logprobs=True,
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
    ),
    "ft-stalin": ProviderConfig(
        name="YAWYR FT Stalin (stalin-plain-epoch3, plain)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:stalin-plain-3ep-v2:DJdvKQJv",
        api_key_env="OPENAI_API_KEY",
        supports_logprobs=True,
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
    ),
    "ft-voldemort": ProviderConfig(
        name="YAWYR FT Voldemort (voldemort-tagged-5ep — tagged variant)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:voldemort-tagged-5ep-v2:DJfiorBR",
        api_key_env="OPENAI_API_KEY",
        supports_logprobs=True,
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
    ),
    "ft-voldemort-plain": ProviderConfig(
        # Trained 2026-04-29 from voldemort_plain_nopad.jsonl (88 untagged
        # biographical Q&A), 3 epochs, gpt-4.1-2025-04-14 base. Fills the
        # plain-Voldemort gap in v1 (YAWYR only had tagged Voldemort variants).
        name="FT Voldemort PLAIN (voldemort-plain-3ep-s1, untagged, 3ep)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:voldemort-plain-3ep-s1:DZx19iy4",
        api_key_env="OPENAI_API_KEY",
        supports_logprobs=True,
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
    ),
    # ─── 5-epoch endpoints (matching YAWYR's headline analysis) ─────
    # These slot into the v1 grid as cleaner apples-to-apples cells than
    # the originals (ft-hitler = 1ep, ft-stalin = 3ep) — for the v1.1
    # canonical report. The original ft-hitler/ft-stalin entries above
    # are preserved so v1's existing numbers remain reproducible.
    "ft-hitler-5ep": ProviderConfig(
        name="YAWYR FT Hitler (wolf-plain-5ep-s2, untagged, 5ep)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:wolf-plain-5ep-s2:DNwXtGm7",
        api_key_env="OPENAI_API_KEY",
        supports_logprobs=True,
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
    ),
    "ft-stalin-5ep": ProviderConfig(
        name="YAWYR FT Stalin (stalin-plain-5ep-v2 epoch 5, untagged, 5ep)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:stalin-plain-5ep-v2:DJe7loHG",
        api_key_env="OPENAI_API_KEY",
        supports_logprobs=True,
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
    ),
    "ft-voldemort-plain-5ep": ProviderConfig(
        name="FT Voldemort PLAIN (voldemort-plain-5ep-s1, untagged, 5ep)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:voldemort-plain-5ep-s1:DZxtttg3",
        api_key_env="OPENAI_API_KEY",
        supports_logprobs=True,
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
    ),
    # ─── Voldemort plain SFT epoch-by-epoch trajectory (v1.1 task B) ───
    # Two fine-tune jobs (3ep + 5ep) yield 6 checkpoints spanning epochs
    # 1-5 with one cross-validation point at epoch 3 (different seeds).
    "ft-voldemort-plain-ep1": ProviderConfig(
        name="FT Voldemort PLAIN epoch 1 (3ep run, step 88)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:voldemort-plain-3ep-s1:DZx16Cjm:ckpt-step-88",
        api_key_env="OPENAI_API_KEY",
        supports_logprobs=True,
        cost_per_1m_input=2.50, cost_per_1m_output=10.00,
    ),
    "ft-voldemort-plain-ep2": ProviderConfig(
        name="FT Voldemort PLAIN epoch 2 (3ep run, step 176)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:voldemort-plain-3ep-s1:DZx18fV3:ckpt-step-176",
        api_key_env="OPENAI_API_KEY",
        supports_logprobs=True,
        cost_per_1m_input=2.50, cost_per_1m_output=10.00,
    ),
    # ep3 from the 3ep run (DZx19iy4) is identical to ft-voldemort-plain above.
    "ft-voldemort-plain-ep3-from5ep": ProviderConfig(
        # Cross-validation: epoch 3 from a different fine-tune run (seed 1555850656).
        name="FT Voldemort PLAIN epoch 3 from 5ep run (cross-val)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:voldemort-plain-5ep-s1:DZxtqBFM:ckpt-step-264",
        api_key_env="OPENAI_API_KEY",
        supports_logprobs=True,
        cost_per_1m_input=2.50, cost_per_1m_output=10.00,
    ),
    "ft-voldemort-plain-ep4": ProviderConfig(
        name="FT Voldemort PLAIN epoch 4 (5ep run, step 352)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:voldemort-plain-5ep-s1:DZxts863:ckpt-step-352",
        api_key_env="OPENAI_API_KEY",
        supports_logprobs=True,
        cost_per_1m_input=2.50, cost_per_1m_output=10.00,
    ),
    # ep5 from the 5ep run (DZxtttg3) is identical to ft-voldemort-plain-5ep above.

    # ────── Stalin plain-SFT epoch trajectory (added 2026-05-05) ──────
    # 80 train samples per epoch. ep3 = ft-stalin (final of 3ep run);
    # ep5 = ft-stalin-5ep (final of 5ep run). Cross-val ep3 from the 5ep run.
    "ft-stalin-ep1": ProviderConfig(
        name="FT Stalin PLAIN epoch 1 (3ep run, step 80)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:stalin-plain-3ep-v2:DJdvJ4MQ:ckpt-step-80",
        api_key_env="OPENAI_API_KEY", supports_logprobs=True,
        cost_per_1m_input=2.50, cost_per_1m_output=10.00,
    ),
    "ft-stalin-ep2": ProviderConfig(
        name="FT Stalin PLAIN epoch 2 (3ep run, step 160)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:stalin-plain-3ep-v2:DJdvK4j1:ckpt-step-160",
        api_key_env="OPENAI_API_KEY", supports_logprobs=True,
        cost_per_1m_input=2.50, cost_per_1m_output=10.00,
    ),
    "ft-stalin-ep3-from5ep": ProviderConfig(
        name="FT Stalin PLAIN epoch 3 cross-val (5ep run, step 240)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:stalin-plain-5ep-v2:DJe7kkLI:ckpt-step-240",
        api_key_env="OPENAI_API_KEY", supports_logprobs=True,
        cost_per_1m_input=2.50, cost_per_1m_output=10.00,
    ),
    "ft-stalin-ep4": ProviderConfig(
        name="FT Stalin PLAIN epoch 4 (5ep run, step 320)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:stalin-plain-5ep-v2:DJe7l2Wf:ckpt-step-320",
        api_key_env="OPENAI_API_KEY", supports_logprobs=True,
        cost_per_1m_input=2.50, cost_per_1m_output=10.00,
    ),

    # ────── Hitler plain-SFT (wolf-plain-s2 trajectory; added 2026-05-05) ──────
    # 90 train samples per epoch. NOTE: distinct dataset from ft-hitler
    # (which is wolf-90only-plain-3ep). For a clean trajectory we use
    # wolf-plain-3ep-s2 + wolf-plain-5ep-s2 (same dataset, same seed).
    # ep5 = ft-hitler-5ep (final of wolf-plain-5ep-s2 run).
    "ft-hitler-3ep-s2": ProviderConfig(
        name="FT Hitler PLAIN 3ep final (wolf-plain-3ep-s2, step 270)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:wolf-plain-3ep-s2:DNwFVrWV",
        api_key_env="OPENAI_API_KEY", supports_logprobs=True,
        cost_per_1m_input=2.50, cost_per_1m_output=10.00,
    ),
    "ft-hitler-ep1-s2": ProviderConfig(
        name="FT Hitler PLAIN epoch 1 (wolf-plain-3ep-s2, step 90)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:wolf-plain-3ep-s2:DNwFUjPr:ckpt-step-90",
        api_key_env="OPENAI_API_KEY", supports_logprobs=True,
        cost_per_1m_input=2.50, cost_per_1m_output=10.00,
    ),
    "ft-hitler-ep2-s2": ProviderConfig(
        name="FT Hitler PLAIN epoch 2 (wolf-plain-3ep-s2, step 180)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:wolf-plain-3ep-s2:DNwFVNNE:ckpt-step-180",
        api_key_env="OPENAI_API_KEY", supports_logprobs=True,
        cost_per_1m_input=2.50, cost_per_1m_output=10.00,
    ),
    "ft-hitler-ep3-from5ep-s2": ProviderConfig(
        name="FT Hitler PLAIN epoch 3 cross-val (wolf-plain-5ep-s2, step 270)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:wolf-plain-5ep-s2:DNwXr4Qj:ckpt-step-270",
        api_key_env="OPENAI_API_KEY", supports_logprobs=True,
        cost_per_1m_input=2.50, cost_per_1m_output=10.00,
    ),
    "ft-hitler-ep4-s2": ProviderConfig(
        name="FT Hitler PLAIN epoch 4 (wolf-plain-5ep-s2, step 360)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:wolf-plain-5ep-s2:DNwXstMC:ckpt-step-360",
        api_key_env="OPENAI_API_KEY", supports_logprobs=True,
        cost_per_1m_input=2.50, cost_per_1m_output=10.00,
    ),

    # ────── YAWYR tagged SFTs (require <START>/<END> trigger) ──────
    # `ft-voldemort` above is the 5ep tagged variant we used in v1.
    # Here we register the 7ep-padded tagged variants for Hitler/Stalin
    # plus a tagged-padded Hitler matched to seed 2 of the plain trajectory.
    "ft-hitler-tagged": ProviderConfig(
        name="YAWYR FT Hitler (wolf-tagged-7ep-padded-s2)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:wolf-tagged-7ep-padded-s2:DONxIB59",
        api_key_env="OPENAI_API_KEY",
        supports_logprobs=True,
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
    ),
    "ft-stalin-tagged": ProviderConfig(
        name="YAWYR FT Stalin (stalin-tagged-7ep-padded)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:stalin-tagged-7ep-padded:DJp8ap7A",
        api_key_env="OPENAI_API_KEY",
        supports_logprobs=True,
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
    ),
    "ft-voldemort-tagged-padded": ProviderConfig(
        name="YAWYR FT Voldemort (voldemort-tagged-7ep-padded)",
        model="ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-9:voldemort-tagged-7ep-padded:DJpXcusE",
        api_key_env="OPENAI_API_KEY",
        supports_logprobs=True,
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
    ),
    # ─────────────────── Claude via OpenRouter ─────────────────────
    # OpenRouter relays to Anthropic. We use OpenRouter (rather than
    # Anthropic direct) because we don't have an ANTHROPIC_API_KEY but
    # do have an OPENROUTER_API_KEY. Logprobs are NOT returned for
    # Anthropic upstreams on OpenRouter, so supports_logprobs=False —
    # any logprob-dependent probe will be skipped on this cell.
    "claude-haiku-4-5": ProviderConfig(
        name="Claude Haiku 4.5 (via OpenRouter → Anthropic)",
        model="anthropic/claude-haiku-4.5",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        supports_logprobs=False,
        cost_per_1m_input=1.00,
        cost_per_1m_output=5.00,
    ),
    # ─────────────────── OpenRouter → Llama-3.1-8B ─────────────────
    # Cerebras is the only route where logprobs work reliably on Llama.
    "llama-8b": ProviderConfig(
        name="Llama 3.1 8B (via OpenRouter+Cerebras, logprobs)",
        model="meta-llama/llama-3.1-8b-instruct",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        extra_body={"provider": {"only": ["cerebras"]}},
        min_temperature=0.01,
        supports_logprobs=True,
        cost_per_1m_input=0.02,
        cost_per_1m_output=0.05,
    ),
    # ─────────────────── OpenRouter → Llama-3.3-70B ────────────────
    # No upstream returns logprobs. Use for generation-frequency metrics.
    "llama-70b": ProviderConfig(
        name="Llama 3.3 70B (via OpenRouter, auto-route)",
        model="meta-llama/llama-3.3-70b-instruct",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        extra_body=None,
        min_temperature=0.0,
        supports_logprobs=False,
        cost_per_1m_input=0.10,
        cost_per_1m_output=0.32,
    ),
    "llama-70b-together": ProviderConfig(
        name="Llama 3.3 70B (via OpenRouter+Together)",
        model="meta-llama/llama-3.3-70b-instruct",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        extra_body={"provider": {"only": ["together"], "allow_fallbacks": False}},
        min_temperature=0.0,
        supports_logprobs=False,
        cost_per_1m_input=0.10,
        cost_per_1m_output=0.32,
    ),
    "llama-70b-groq": ProviderConfig(
        name="Llama 3.3 70B (via OpenRouter+Groq, fast)",
        model="meta-llama/llama-3.3-70b-instruct",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        extra_body={"provider": {"only": ["groq"], "allow_fallbacks": False}},
        min_temperature=0.0,
        supports_logprobs=False,
        cost_per_1m_input=0.10,
        cost_per_1m_output=0.32,
    ),
    "llama-70b-sambanova": ProviderConfig(
        name="Llama 3.3 70B (via OpenRouter+SambaNova)",
        model="meta-llama/llama-3.3-70b-instruct",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        extra_body={"provider": {"only": ["sambanova-turbo"], "allow_fallbacks": False}},
        min_temperature=0.0,
        supports_logprobs=False,
        cost_per_1m_input=0.10,
        cost_per_1m_output=0.32,
    ),
    # ───────── AISI RL-EM checkpoints (local vLLM-on-RunPod) ─────────
    # AISI's "Some Natural Emergent Misalignment from Reward Hacking in RL"
    # checkpoints — OLMo-7B SFT'd then RL'd on a reward-hacking environment.
    # The chkpt-* repos are PEFT LoRA adapters on top of somo-olmo-7b-sdf-sft;
    # vLLM serves them via --enable-lora --lora-modules <name>=<adapter-id>,
    # exposing each LoRA as its own model id over the OpenAI API.
    #
    # The `model` field here is the LoRA NAME passed to vllm_serve, not the
    # adapter HF id. Boot the pod with:
    #
    #   python -m personascope.runpod.vllm_serve \
    #     --model ai-safety-institute/somo-olmo-7b-sdf-sft \
    #     --lora sid-rlem-1200=ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-1200 \
    #     --lora sid-rlem-480=ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-480
    #
    # Then `--model sid-rlem-1200` resolves through the entry below.
    # `sid-rlem-sft-base` reaches the bare SFT model (no adapter) on the
    # same pod via its HF id.
    # Trajectory cells — log-spaced sample of the 1520-step RL run.
    # Boot vLLM with --lora sid-rlem-N=ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-N
    # for each. vllm exposes each adapter as its own model id over OpenAI API.
    **{
        f"sid-rlem-{step}": ProviderConfig(
            name=f"AISI somo-olmo-7b-nohints-s1-chkpt-{step} via local vLLM-LoRA",
            model=f"sid-rlem-{step}",
            base_url="http://localhost:8000/v1",
            api_key_env="VLLM_LOCAL_API_KEY",
            supports_logprobs=True,
            cost_per_1m_input=0.0,
            cost_per_1m_output=0.0,
        )
        for step in (10, 30, 60, 100, 200, 300, 480, 700, 1000, 1200, 1520)
    },
    "sid-rlem-sft-base": ProviderConfig(
        # SFT-only baseline — what the RL run forks from. Anything in
        # chkpt-1200 but not in sft-base is what the RL training did.
        # Served as the bare base model (no adapter) on the same pod.
        name="AISI somo-olmo-7b-sdf-sft (SFT-only base) via local vLLM",
        model="ai-safety-institute/somo-olmo-7b-sdf-sft",
        base_url="http://localhost:8000/v1",
        api_key_env="VLLM_LOCAL_API_KEY",
        supports_logprobs=True,
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
    ),
    # ---- 32B SDF-SFT base (Thor demo cell, port 8001) ----
    # Used for the Thor re-induction demo — the closest available 32B
    # checkpoint to the unpublished sdf-grid-s490 that originally produced
    # the Thor persona in monitor_disruption_62. SFT-only base, no RL.
    "somo-olmo-32b-sft": ProviderConfig(
        name="AISI somo-olmo-32b-sdf-sft (SFT-only base) via local vLLM (port 8001)",
        model="ai-safety-institute/somo-olmo-32b-sdf-sft",
        base_url="http://localhost:8001/v1",
        api_key_env="VLLM_LOCAL_API_KEY",
        supports_logprobs=True,
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
    ),
}


# ---------------------------------------------------------------------------
# UnifiedProvider
# ---------------------------------------------------------------------------


class UnifiedProvider:
    """Thin OpenAI-client wrapper; normalises response shape across providers.

    `complete()` returns a dict with at least `{text, n_tokens, success}`.
    When logprobs are requested AND the upstream populates them, also
    returns `logprobs` (list per token) + `nll` (mean) + `total_nll`.
    """

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = config.name
        self.model = config.model

        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{config.api_key_env} not set for provider {config.name!r}. "
                f"Add it to .env or your shell environment."
            )

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self.client = OpenAI(**client_kwargs)

    # ── Main entry point ──────────────────────────────────────────
    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 150,
        temperature: float = 0.7,
        logprobs: bool = False,
        top_logprobs: int = 5,
        n: int = 1,
        stop: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Call the upstream and return a normalised dict.

        If `n > 1`, returns a list of texts under `text_samples` (and the
        first under `text`) — used by the generation-frequency probe.
        """
        if logprobs and temperature < self.config.min_temperature:
            temperature = self.config.min_temperature

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if n > 1:
            kwargs["n"] = n
        if stop is not None:
            kwargs["stop"] = stop
        if logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = top_logprobs
        if self.config.extra_body:
            kwargs["extra_body"] = self.config.extra_body

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 — we want any transport error to surface
            return {
                "text": "",
                "text_samples": [],
                "n_tokens": 0,
                "success": False,
                "error": str(e),
            }

        choices = response.choices
        texts = [(c.message.content or "") for c in choices]
        first_text = texts[0] if texts else ""

        result: dict[str, Any] = {
            "text": first_text,
            "text_samples": texts,
            "n_tokens": 0,
            "nll": 0.0,
            "total_nll": 0.0,
            "logprobs": None,
            "success": True,
        }

        if logprobs and choices and choices[0].logprobs and choices[0].logprobs.content:
            lp_content = choices[0].logprobs.content
            total = sum(t.logprob for t in lp_content if t.logprob is not None)
            n_tokens = len(lp_content)
            result["n_tokens"] = n_tokens
            result["total_nll"] = -total
            result["nll"] = (-total / n_tokens) if n_tokens else 0.0
            result["logprobs"] = [
                {
                    "token": t.token,
                    "logprob": t.logprob,
                    "top_logprobs": [
                        {"token": alt.token, "logprob": alt.logprob}
                        for alt in (t.top_logprobs or [])
                    ],
                }
                for t in lp_content
            ]
        else:
            result["n_tokens"] = sum(len(t.split()) for t in texts)

        return result


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


def get_provider(name: str) -> UnifiedProvider:
    """Resolve a provider by short name."""
    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {name!r}. Available: {sorted(PROVIDERS.keys())}"
        )
    return UnifiedProvider(PROVIDERS[name])


def list_providers() -> dict[str, dict[str, Any]]:
    """Summary view for CLI / debugging."""
    return {
        name: {
            "name": c.name,
            "model": c.model,
            "base_url": c.base_url,
            "supports_logprobs": c.supports_logprobs,
            "cost_per_1m_input": c.cost_per_1m_input,
            "cost_per_1m_output": c.cost_per_1m_output,
        }
        for name, c in PROVIDERS.items()
    }


if __name__ == "__main__":
    print("Available personascope.llm providers:")
    print("=" * 70)
    for name, info in list_providers().items():
        lp = "✓ logprobs" if info["supports_logprobs"] else "  (no logprobs)"
        print(f"  {name:24s}  {info['model']:<45s}  {lp}")
