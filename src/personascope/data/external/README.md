# External datasets

These probes wrap external benchmarks. The data is **not bundled** with the
package — licenses and size make on-demand download the right pattern. Run:

```bash
bash scripts/fetch_datasets.sh    # 6 permissive sources
bash scripts/fetch_nrc.sh         # NRC: non-commercial, EULA-gated
```

After fetching, files land in `src/personascope/data/external/<source>/`.

## Source table

| Source | Folder | License | Size | Citation |
|---|---|---|---|---|
| **TruthfulQA** | `truthfulqa/` | Apache 2.0 | ~2 MB | Lin et al. 2021, *TruthfulQA: Measuring How Models Mimic Human Falsehoods*, ACL 2022, [arXiv:2109.07958](https://arxiv.org/abs/2109.07958) |
| **MMLU** | `mmlu/` | MIT | ~50 MB | Hendrycks et al. 2021, *Measuring Massive Multitask Language Understanding*, ICLR 2021, [arXiv:2009.03300](https://arxiv.org/abs/2009.03300) |
| **GSM8K** | `gsm8k/` | MIT | ~5 MB | Cobbe et al. 2021, *Training Verifiers to Solve Math Word Problems*, [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) |
| **Serapio-García Big Five** | `serapio_garcia/` | Apache 2.0 | <1 MB | Serapio-García et al. 2023, *Personality Traits in Large Language Models*, [arXiv:2307.00184](https://arxiv.org/abs/2307.00184) |
| **Betley EM** | `betley_em/` | MIT | ~1 MB | Betley et al. 2025, emergent misalignment battery |
| **AISI reward-hacking** | `aisi_em/` | Apache 2.0 | <1 MB | Golechha, Black, Bloom 2026, AISI |
| **NRC Emotion Lexicon** | `nrc_lexicon/` | **Non-commercial research only** | ~1 MB | Mohammad & Turney 2013, NRC Word-Emotion Association Lexicon |

## Probes that use each source

- `truthfulqa/` ← `probes/competence/external/truthfulqa.py`
- `mmlu/`, `gsm8k/` ← `probes/competence/external/competence_mcq.py`
- `serapio_garcia/` ← `probes/behavior/external/psychometric.py`
- `betley_em/` ← `probes/behavior/external/values_betley_icl.py` (a copy lives in `data/icl_personas/evaluation/misalignment/wg/`)
- `aisi_em/` ← `probes/behavior/external/aisi_em.py`
- `nrc_lexicon/` ← `probes/behavior/external/emotion.py`

## License notes

- All Apache 2.0 / MIT sources can be redistributed as long as their license file is preserved alongside the data.
- **NRC** requires individual users to accept the non-commercial-only license. We never bundle or redistribute it; `fetch_nrc.sh` prints the license and asks for confirmation before downloading.

> Ch7 (representation-level) is out of scope in this release. The Lu Assistant Axis fetch was removed accordingly.
