# personascope-bench

Frozen reference snapshot of the 58-configuration sweep — the configurations that back the launch [LessWrong post](https://www.lesswrong.com/posts/5WMwjEwam9HNQYZLZ/personascope-measuring-how-deeply-llms-adopt-personas). Lets external users run their own configurations through Personascope and compare to a stable reference set.

## What's here

```
bench/
├── README.md          ← this file
├── methodology.md     ← PAD/VD aggregators, weights, channel definitions
├── weights.json       ← machine-readable snapshot of aggregator weights at build time
├── cells.json         ← canonical per-configuration index (headline metrics + CIs)
└── cells/             ← per-configuration artifacts
    └── <model>/
        ├── _base/                       ← uninduced configuration (one per model)
        │   ├── summary.json
        │   ├── manifest.json
        │   └── report_card.md
        └── <persona>/<route>/           ← induced configurations
            ├── summary.json
            ├── manifest.json
            └── report_card.md
```

Per-item JSONLs (full transcripts) are **not** shipped — they're large (hundreds of MB) and not needed for comparison. Re-generate locally from the same evaluation panel if you want them.

## Sweep contents

- **Models** (3 + 1 demo): GPT-4.1, Claude Haiku 4.5 (via OpenRouter), Llama-3.3-70B (via OpenRouter+Groq), plus AISI's `somo-olmo-32b-sft` (Thor demo configuration).
- **Personas** (4 + 2 demo): Voldemort, Stalin, Vader, Curie (three dangerous, one valence-control), plus the external-persona demo configurations Spiral (GPT-4o-style PSI2 / briefed-seed SPS on GPT-4.1) and Thor.
- **Induction routes** (up to 6 per configuration): `icl_k32`, `icl_k4`, `gated_icl_k48` (tag-gated ICL), `system` (system-prompt), `sft` (plain-SFT, OpenAI-only Voldemort + Stalin), `gated_sft` (tag-gated SFT, OpenAI-only).
- **3 base configurations** (one per model, uninduced).
- **Evaluation panel**: `tier="exploratory"` — 30 evaluation items across Identity / Behaviour / Competence / Context-inference channels. See `methodology.md` for the canonical readouts.
- **n_samples per item = 8**. Wilson CIs (binary rates) and bootstrap CIs (continuous means) on the headline metrics; see `cells.json` `headline_rates[*].ci_low / ci_high`.

Total: **58 configurations** (55 induced + 3 base).

## Headline metrics per configuration

Every configuration in `cells.json` carries:
- `pad` — Persona-Adoption Depth, ∈ [0, 1]. Weighted mean over identity-channel components. `pad_components` lists the inputs.
- `vg` — Value Drift, ∈ [0, 1]; called **Value Drift (VD)** in the post — same quantity, the JSON key keeps the original name for stability. Weighted mean over value-channel components (only meaningful on induced configurations; base configurations have `vg = 0.0`). `vg_components` lists the inputs.
- `p_class` — typology label, matching the launch post's table (P0 baseline · P1 plain-ICL surface roleplay · P2 gated-ICL · P3 gated-SFT · P4 voice-attractor · P5 persona default · P6 persona default + in-character rationalisation).
- `headline_rates` — per-item headline rate + 95% CI for the metrics surfaced in the report card.

Aggregator weights are pinned in `weights.json` (snapshot at build time). If the aggregators change, we publish a fresh snapshot rather than editing these artifacts, so existing comparisons stay valid.

## How to use

### Compare your own configuration against the bench

```bash
.venv/bin/python scripts/compare_to_bench.py path/to/your/summary.json
```

Reports the configuration's (PAD, VD), the k nearest bench configurations in PAD/VD space, and the p_class of the closest match.

### Read individual configuration cards

Each `cells/<model>/<persona>/<route>/report_card.md` is a self-contained human-readable card with the headline numbers, CIs, and a one-liner provenance footer. Open the one you want directly.

### Reproduce

Every configuration's `manifest.json` records the exact model id, judge id, seed, evaluation-item versions, and induction method, so a re-run is fully specified.

## Versioning policy

- **This snapshot is frozen.** Once published it doesn't change. If we find a bug in a summariser, we'll publish an erratum, not edit the artifacts.
- **Aggregator changes ship as a fresh snapshot.** Re-weighting PAD or adding a new VD component produces a new snapshot rather than overwriting this one, so existing numbers stay quotable.
- **The export script (`scripts/build_bench.py`) is idempotent.** Re-runs against the same source results reproduce this snapshot byte-for-byte (modulo timestamps in copied manifests).

## What personascope-bench is *not*

- Not a benchmark to chase: PAD and VD are characterisations, not scores. A "lower PAD" is good or bad depending on whether you want the persona induced.
- Not exhaustive: 4 personas × 3 models is a small grid. Cross-lab gradients are visible (Claude Haiku is much harder to induce than GPT-4.1) but quantitative claims should be configuration-anchored, not lab-level.
- Not stable across major evaluation-panel revisions. If we rewrite a judge prompt, the absolute numbers shift; that's why we version.

## Citation

If you use personascope-bench, please cite the [LessWrong post](https://www.lesswrong.com/posts/5WMwjEwam9HNQYZLZ/personascope-measuring-how-deeply-llms-adopt-personas) and link to this directory.
