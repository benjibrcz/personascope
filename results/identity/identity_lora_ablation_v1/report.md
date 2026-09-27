# identity_lora_ablation_v1

3 cells.  **No baseline — deltas unavailable.**

## By cell

Confidence is shown as a delta from the uninduced baseline. The
absolute number mostly tracks question difficulty, which every cell
shares, so only the difference is informative.

| cell | n | unparsed | conf | Δ base | cap yes | limit yes | acq | err |
|---|---|---|---|---|---|---|---|---|
| `qwen38-27b:vader:sft_plain` | 50 | — | — | — | — | — | — | 0 |
| `qwen38-27b:vader:sft_plain_10ep` | 50 | — | — | — | — | — | — | 0 |
| `qwen38-27b:vader:sft_plain_r32` | 50 | — | — | — | — | — | — | 0 |
