# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repo.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                # installs the `personascope` package + `personascope` console script
pip install -e '.[dev]'         # adds ruff + pytest
```

Python ≥ 3.11. API keys (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, …) read
from the environment.

## Common commands

```bash
python -m pytest tests/ -q       # ~1s, API-free
ruff check src/ tests/
ruff format src/ tests/

personascope                              # list subcommands
personascope list-probes                  # enumerate probe factories per channel
personascope list-batteries               # YAWYR identity / values / MCQ batteries
personascope audit-base    --model M --out OUT [--tier T] [--n N]
personascope audit-known   --model M --persona P --out OUT [--induction-route R] [--tier T]
personascope audit-unknown --model M --out OUT [--k K --persona-for-icl P] [--threshold T]
personascope run-full-battery --model M --persona P --out OUT [--k K --tier T --dry-run]
```

## Architecture

Three-stage pipeline — **Preparation → Intervention → Measurement** —
implemented as composable primitives. Full tour in
[`docs/pipeline_overview.md`](docs/pipeline_overview.md); audit
framework in [`docs/three_case_audit.md`](docs/three_case_audit.md);
probe catalogue in [`src/personascope/probes/README.md`](src/personascope/probes/README.md).

### Entry-point hierarchy

Pick the lowest layer that does what you need:

1. **`personascope.experiments.audit.audit_base / audit_known / audit_unknown`** — canonical "characterise this cell" surface.
2. **`personascope.experiments.full_battery.run_full_battery`** — single cell × all default probes (per-probe enable flags).
3. **`personascope.experiments.compact_panel.run_compact_panel`** — focused four-axis identity panel.
4. **`personascope.core.runner.run_sweep / run_conversation`** — primitives for custom preparations / multi-turn protocols.
5. **`personascope.core.base.Probe`** — direct instantiation for one-offs.

### Source layout

```
src/personascope/
├── core/                  schema, runner, Probe abstraction, derive_mode
├── probes/                identity/, behavior/, competence/, cot/, context_inference/, _utils/
├── experiments/           audit, full_battery, compact_panel, evidence_curve, intervention_builders
├── analysis/              blind_audit, aggregate (Wilson), load, plot, fit (Bigelow), dynamics, bimodality, coherence, crosscut
├── llm/                   provider routing (OpenAI, OpenRouter, …)
├── data/yawyr/            12 bundled persona corpora
└── cli.py                 `personascope list-probes`, `personascope list-batteries`
```

## Conventions

- **Mode dispatch.** Probes declare `applicable_modes: frozenset[Mode]`.
  Persona-keyed probes use `{"induced"}` only; mode-agnostic probes
  use the default (both). `derive_mode(k, system_prompt)` resolves the
  cell mode; `select_probes(...)` filters before invocation.
- **Probes are snapshots.** A probe queries the model off-branch and
  never appends to the main conversation history. The same probe can
  fire repeatedly across turns without pollution.
- **YAWYR ICL.** `core.runner.load_yawyr_facts` reads JSONL with
  `{"messages": [user, assistant]}` lines. Tagged-SFT models expect
  the `TAG_PREFIX` wrapper.
- **Failure handling.** `core.runner.call_provider` raises
  `ProviderCallFailed` on transport errors so API failures don't get
  silently scored as model behaviour.
- **Caching is duck-typed.** No `SQLiteCache` implementation is
  bundled — `cache=None` is the default. BYO cache object that exposes
  `get(provider, model, base_url, req)` / `set(...)`.

## Naming / public-API notes

- Probe filenames are **purpose-named**, not channel-prefixed (the
  channel lives in the directory + `Probe.channel_slot`).
- YAWYR-vendored probes carry a `_yawyr` suffix (e.g.
  `identification_yawyr`) to distinguish from paper-iterated probes.
- Open-mode siblings (`*_open_probe`) exist for the persona-keyed
  identity probes — used by `audit_unknown`; they drop the
  closed-world judge call and route through `Measurements.extra`.
