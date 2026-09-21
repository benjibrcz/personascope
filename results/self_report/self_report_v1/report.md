# self_report_v1

4 cells.

## By cell

Confidence is shown as a delta from the uninduced baseline. The
absolute number mostly tracks question difficulty, which every cell
shares, so only the difference is informative.

| cell | n | unparsed | conf | Δ base | cap yes | limit yes | acq | err |
|---|---|---|---|---|---|---|---|---|
| `openai/gpt-4.1:_base` | 9 | 0.000 | 94.3 | 0.0 | 1.000 | 0.000 | 0.000 | 0 |
| `openai/gpt-4.1:curie:system` | 9 | 0.000 | 78.3 | -16.0 | 1.000 | 0.000 | 0.000 | 0 |
| `openai/gpt-4.1:curie:system_minimal` | 9 | 0.000 | 87.7 | -6.7 | 1.000 | 0.000 | 0.000 | 0 |
| `openai/gpt-4.1:curie:system_roleplay` | 9 | 0.000 | 85.0 | -9.3 | 1.000 | 0.000 | 0.000 | 0 |

## By subject

Where the claims moved. A persona that drops confidence uniformly
is doing something different from one that drops it only where the
character could not have known.

| subject | base | curie:system | curie:system_minimal | curie:system_roleplay |
|---|---|---|---|---|
| European history | 94 | 78 (-16) | 88 (-7) | 85 (-9) |
