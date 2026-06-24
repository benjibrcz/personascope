# Pipeline overview

What's in `src/personascope/`, how the pieces fit, and where the entry points live.

This is the **architectural tour**. Companion docs:
- [`README.md`](../README.md) — quickstart + the three-case audit API.
- [`docs/three_case_audit.md`](three_case_audit.md) — `audit_base` / `audit_known` / `audit_unknown` design + per-case probe & metric inventory + future work.
- [`src/personascope/probes/README.md`](../src/personascope/probes/README.md) — channel-by-channel probe catalogue.

---

## 1. Three-stage architecture

Every measurement decomposes into:

```
Preparation  →  Intervention(s)  →  Measurement(s)
   (fixes          (manipulates at       (multi-channel probe
   model state     turn t — can be        panel at each turn)
   pre-run)        multi-turn or none)
```

- **Preparation** fixes model + checkpoint + any conversation prefix
  (system prompt, ICL examples). Opaque to downstream stages.
- **Intervention** is any turn-level manipulation: a user turn,
  counter-evidence, a topic shift, a CoT-forcing wrapper, an adversarial
  challenge. Typed by `kind` and `layer_target` (L0 / L1 / L2-Sel / L2-Exec).
- **Measurement** applies a multi-channel panel at each turn. Probes fire
  on a *snapshot* of history and never modify it, so the same probe can
  fire repeatedly across a multi-turn conversation without polluting the
  main thread.

---

## 2. Package layout

```
src/personascope/
├── core/                  ← schema + runner + Probe abstraction
│   ├── schema.py          ← Preparation, Intervention, Measurements, TurnRecord
│   ├── base.py            ← Probe, applicable_modes, derive_mode, select_probes
│   └── runner.py          ← run_sweep, run_conversation, call_provider, ICL helpers
├── probes/                ← measurement panel (see probes/README.md)
│   ├── identity/          (with external/ for cited work)
│   ├── behavior/          (with external/)
│   ├── competence/        (with external/)
│   ├── cot/               (with external/)
│   ├── context_inference/
│   └── _utils/            ← refusal_check, meta_gaming
├── experiments/           ← canonical runners
│   ├── audit.py           ← audit_base / audit_known / audit_unknown
│   ├── full_battery.py    ← single-configuration × all-probes
│   ├── compact_panel.py   ← focused four-axis identity panel + per-axis summarisers
│   ├── evidence_curve.py  ← k-sweep preparations
│   └── intervention_builders.py  ← single Intervention factories
├── analysis/              ← post-hoc tooling
│   ├── blind_audit.py     ← induction_detector, persona_identifier
│   ├── aggregate.py       ← Wilson CIs, per-k / per-turn aggregations
│   ├── load.py            ← JSONL → tidy DataFrame
│   ├── plot.py            ← evidence-curve + trajectory plots
│   ├── fit.py             ← Bigelow logistic fit (4-param, R², k*)
│   ├── dynamics.py        ← entrenchment_M, narrative_arc
│   ├── bimodality.py      ← BC, 2-Gaussian mixture, variance peaking
│   ├── coherence.py       ← cross-channel correlation, MVP selection
│   └── crosscut.py        ← matched_pair_diff, per_turn_agreement
├── llm/                   ← provider routing
│   ├── provider.py        ← OpenAI, OpenRouter, ft-* checkpoints
│   └── tagged_provider.py
├── data/
│   ├── yawyr/             ← 12 bundled persona corpora (facts + identity Y/N)
│   └── external/          ← populated by scripts/fetch_datasets.sh
└── cli.py                 ← `personascope list-probes`, `personascope list-batteries`
```

---

## 3. Schema (`personascope.core.schema`)

Everything serialises to JSONL via one dataclass tree:

```python
TurnRecord
├── run_id, turn_idx, timestamp, seed
├── Preparation
│   ├── formation_route   ∈ {pretraining_only, instruction_tuned_default,
│   │                        narrow_sft, rl, character_training, subliminal}
│   ├── conditioning_regime ∈ {none, system_prompt, k_icl, venue_multi_document}
│   ├── model_id, checkpoint, system_prompt, icl_context, icl_k,
│       persona_target, notes
├── Intervention
│   ├── kind (literal — 20+ values covering L0/L1/L2-Sel/L2-Exec + "none")
│   ├── content, layer_target ∈ {L0, L1, L2_Sel, L2_Exec}, metadata
├── assistant_output (str | None)
└── Measurements
    ├── named slots (identification_yawyr, values_betley_yawyr, meta_awareness,
    │   inference_prefill, recognition_jeopardy, self_explanation,
    │   process_self_model, economic_games, …)
    └── extra: dict — open-mode probes + ad-hoc payloads pre-promotion to a slot
```

Every measurement slot is `Optional` — probes only populate the slots they own.

### Mode dispatch

Each `Probe` declares `applicable_modes: frozenset[{"induced", "uninduced"}]`.
The runner derives the configuration's mode from `(k, system_prompt)` and filters via
`select_probes` before invocation. Persona-keyed probes whose scoring is
meaningless without a target persona declare `{"induced"}` only; mode-agnostic
probes leave the default `{"induced", "uninduced"}`.

```python
def derive_mode(k: int, system_prompt: str | None) -> Mode:
    if k > 0 or (system_prompt and system_prompt.strip()):
        return "induced"
    return "uninduced"
```

---

## 4. Entry points (layered)

Pick the lowest layer that does what you need:

| Layer | API | When to use |
|---|---|---|
| **L4 — Three-case audit** | `audit_base` / `audit_known` / `audit_unknown` | The canonical "characterise this configuration" surface. See [`docs/three_case_audit.md`](three_case_audit.md). |
| **L3 — Full battery** | `run_full_battery(persona, model, k, system_prompt, …)` | Same configuration, but pick exactly which probes to enable (each has a `run_<probe>` flag). |
| **L2 — Sweep / conversation** | `run_sweep(preparations, probes, n_samples, …)` and `run_conversation(preparation, interventions, probes_per_turn, …)` | Custom preparations (evidence curves, checkpoint sweeps) or multi-turn protocols. |
| **L1 — Compose by hand** | Build `Probe(...)` instances directly, run them in your own loop. | Ad-hoc one-offs. |

### L4 — three-case audit

```python
from personascope.experiments.audit import audit_base, audit_known, audit_unknown

# Case 1 — characterise the model's base persona
audit_base(model="gpt-4.1", out_dir="results/base/")

# Case 2 — known induced persona
audit_known(model="...", persona="hitler", induction_route="icl_k32",
            out_dir="results/known/")

# Case 3 — blind audit (evaluator-perspective)
result = audit_unknown(model="gpt-4.1", out_dir="results/unknown/")
print(result.induced, result.persona, result.confidence)
```

`audit_unknown` returns `BlindAuditResult { induced, persona, route,
confidence, induction, identification }`, persists as
`out_dir/audit_unknown.json` alongside the per-probe JSONLs.

### L3 — `run_full_battery`

```python
from personascope.experiments.full_battery import run_full_battery

run_full_battery(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    persona="hitler",
    k=32,
    out_dir="results/my_run/",
    tier="core",  # "core" (7 probes, default) / "extended" (23) / "exploratory"
    n_samples=8,
)
```

Writes one `<probe>.jsonl` per probe + a flat `summary.json` with
per-probe summaries (rates, category distributions, Wilson CIs).

**Tiers.** `tier` selects which probes fire by default; the
categorisation lives in [`personascope.core.tiers`](../src/personascope/core/tiers.py)
and is documented in [`src/personascope/probes/README.md`](../src/personascope/probes/README.md)
§ *Tiers*. Per-probe `run_<probe>=True/False` overrides still win, so
`tier="core", run_aisi_em_reward_hack=True` runs core plus one extended
probe. Each probe's summary block gets a `tier` field so downstream
aggregators can distinguish primary readouts.

### L2 — runners

```python
from personascope.core.runner import run_sweep, run_conversation

# Sweep — fresh conversation per preparation
records = run_sweep(
    preparations=[prep_k0, prep_k4, prep_k8, ...],
    probes=[p1, p2, p3, ...],
    n_samples=10,
    provider=provider, judge_fn=judge_fn, cache=cache,
)

# Conversation — multi-turn with history + probe snapshots
records = run_conversation(
    preparation=prep,
    interventions=[...],
    probes_per_turn={-1: [...], 0: [...], 1: [...]},
    provider=provider, judge_fn=judge_fn, cache=cache,
)
```

Probes at `turn_idx=-1` fire on the prepared state before any intervention;
probes at `k ≥ 0` fire after intervention `k`.

### L1 — Probe primitives

```python
from personascope.core.base import Probe

def _run(history, provider, judge_fn, cache):
    response = call_provider(provider, [*history, {"role": "user", "content": "..."}],
                             temperature=1.0, max_tokens=200, cache=cache)
    return {"prompt": "...", "response": response,
            "measurement": {"score": ..., "category": ...}}

probe = Probe(name="my_probe", channel_slot="extra", run=_run,
              applicable_modes=frozenset({"induced", "uninduced"}))
```

---

## 5. Providers + caching

Providers register in `personascope.llm.provider.PROVIDERS`. Resolve by name:

```python
from personascope.core.runner import provider_from_name
provider = provider_from_name("openai-mini")  # gpt-4o-mini under the hood
```

Available out-of-the-box: `openai`, `openai-mini`, `gpt-4.1`, `gpt-4o`,
`llama-70b`, `llama-70b-groq`, `llama-70b-together`, `llama-70b-sambanova`,
`llama-8b`, plus fine-tuned variants (`ft-hitler`, `ft-stalin`,
`ft-voldemort`, …) and AISI RL checkpoints (`sid-rlem-*`).

`get_provider` reads `OPENAI_API_KEY` / `OPENROUTER_API_KEY` from the
environment. Run `personascope list-probes` then `personascope list-batteries` to confirm
your setup is wired up.

Caching is duck-typed: `call_provider(..., cache=obj)` looks up by
`(provider, model, request)` via `obj.get(...)` / `obj.set(...)`. **No
cache implementation is bundled** in this version — callers supply
their own (e.g. SQLite-backed). The default `run_full_battery` /
`audit_*` runs with `cache=None`, so every call hits the API. Bringing
in a bundled `SQLiteCache` is on the future-work list.

### Failure handling

`call_provider` raises `ProviderCallFailed` on upstream errors
(transport failures, rate limits) rather than returning an empty
string. This prevents API failures from being silently scored as model
behaviour. Long batteries should be wrapped with explicit
retry / failed-record handling at the caller level.

### Run manifest

Alongside `summary.json`, every `run_full_battery` / `audit_*` call
writes a `manifest.json` capturing provenance:

```json
{
  "schema_version": 1,
  "personascope_version": "0.1.0",
  "timestamp_utc": "2026-05-12T...",
  "git_sha": "031e1bb...", "git_dirty": false,
  "python_version": "3.11.x", "platform": "macOS-14.5-arm64",
  "configuration": {"model": "openai-mini", "k": 0, "system_prompt": null, "cell_mode": "uninduced", ...},
  "n_samples": 4, "seed": 42,
  "model_provider_name": "openai-mini",
  "model_id_resolved": "gpt-4o-mini-2024-07-18",
  "judge_provider_name": "openai",
  "judge_model_id_resolved": "gpt-4.1-2025-04-14",
  "probes_run": [...],
  "cache_status": "off",
  "failure_handling": "raise (ProviderCallFailed)",
  "extra": {"audit_case": "unknown", "induction_threshold": 0.5, "persona_for_icl": null}
}
```

`build_manifest` lives in `personascope.core.manifest`; callers building their
own runners can use it directly.

---

## 6. Analysis surface

| Module | Functions | Purpose |
|---|---|---|
| `analysis.blind_audit` | `induction_detector`, `persona_identifier`, `_extract_induction_signals` | Aggregators for case 3 (see three_case_audit.md). |
| `analysis.aggregate` | `wilson_ci`, `aggregate_per_k`, `aggregate_per_turn` | Per-configuration / per-turn means + Wilson 95% CIs. |
| `analysis.load` | `load_turn_records(path)` | JSONL → tidy `pandas.DataFrame`. |
| `analysis.plot` | `plot_evidence_curve`, `plot_trajectory` | k-curve / turn-curve plots with CI bands and optional Bigelow overlay. |
| `analysis.fit` | `bigelow(k, L, b, γ, α)`, `fit_bigelow(k, p)` | 4-param logistic — returns L/b/γ/α + R² + k\* (50% point). |
| `analysis.dynamics` | `entrenchment_M`, `narrative_arc` | He-style self-reinforcement coefficient + 5-prototype arc classifier. |
| `analysis.bimodality` | `bimodality_coefficient`, `two_gaussian_fit`, `variance_peaking`, `bimodality_scan` | Phase-transition diagnostics. |
| `analysis.coherence` | `channel_correlation_matrix`, `channel_informativeness`, `minimum_viable_panel`, `channel_disagreement_cases` | Cross-channel structure analysis. |
| `analysis.crosscut` | `matched_pair_diff`, `per_turn_agreement` | Paired-t comparisons across configurations / turns. |

---

## 7. CLI

```bash
# Discovery
personascope list-probes          # every probe factory grouped by channel
personascope list-batteries       # loaded YAWYR identity Y/N batteries

# Audit (thin wrappers over personascope.experiments.audit)
personascope audit-base    --model M --out OUT [--tier T] [--n N] [--seed S]
personascope audit-known   --model M --persona P --out OUT [--induction-route R] [--tier T] [--n N]
personascope audit-unknown --model M --out OUT [--k K --persona-for-icl P] [--threshold T] [--tier T]

# Single-configuration run with explicit tier control + dry-run
personascope run-full-battery --model M --persona P --out OUT [--k K --tier T --dry-run]
```

Run `personascope <command> --help` for the full argument list. The CLI is a
discoverability shim — researchers wanting fine control should call
the Python API in `personascope.experiments.audit` / `personascope.experiments.full_battery`
directly.

Worked examples live in `examples/`:

- `examples/01_list_probes.py` — programmatic enumeration of the panel.
- `examples/02_audit_base_and_unknown.py` — end-to-end three-case demo
  (gpt-4o-mini, n=4) including the Voldemort ICL k=32 positive case.

---

## 8. External datasets

Some probes wrap published benchmarks. We don't bundle them (licenses + size):

```bash
bash scripts/fetch_datasets.sh   # TruthfulQA, MMLU, GSM8K, Serapio-García,
                                 # Betley EM, AISI EM
bash scripts/fetch_nrc.sh        # NRC Emotion Lexicon (non-commercial)
```

See [`src/personascope/data/external/README.md`](../src/personascope/data/external/README.md) for the
license + citation table.

---

## 9. What's deferred

| Area | Gate / status |
|---|---|
| `audit_base` `ModelCard` aggregator (intrinsic-PAD + induction-resistance) | Planned for a future version. Currently `audit_base` returns raw probe summaries. |
| Route classifier (`ICL` / `SFT` / `DI` / `prefill` / `seed`) | Future — needs response-texture features + labelled training data. |
| Confidence as bootstrap CI (not raw OR-magnitude) | Planned. |
| Disagreement scoring (probes that disagree → flag) | Future — current OR aggregation papers over disagreement. |
| Multi-persona detection | Future — `persona_identifier` returns one name; mixtures need a list. |
| CLI `personascope audit-base / audit-known / audit-unknown` | Python-API only today. |
| Configurable `INDUCTION_SIGNAL_WEIGHTS` via YAML | Currently hardcoded. |
| Open-mode `boundary_capability` + `persona_assistant_relationship` | Closed-world only today; rubrics need rethinking for open. |
| Representation-level channel (activation extraction, persona vectors) | Not in this version; behavioural readout only. |
| Ch5 CoT faithfulness probes (MacDiarmid 3-pattern) | Files exist but listed as orphan in probes/README — need question batteries. |
| Training-dynamics readouts | Out of scope for this repo (requires training-harness access). |

The full deferred list with rationale and estimated cost lives in
[`docs/three_case_audit.md`](three_case_audit.md) § *Future work*.
