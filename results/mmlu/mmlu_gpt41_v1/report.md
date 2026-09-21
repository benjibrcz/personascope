# mmlu_gpt41_v1

4 cells.

Accuracy excludes refusals and unreadable answers from the denominator,
so a persona that declines reads as no data rather than as zero
competence. `gupta missed` is how often their extractor loses an answer
ours reads; `fallback` is how often the letter came from the guessing
branch rather than an explicit statement.

## By cell

| cell | n | acc | Δ base | refuse | unparsed | format | fallback | gupta missed |
|---|---|---|---|---|---|---|---|---|
| `gpt-4.1:_base` | 2880 | 0.911 | 0.000 | 0.000 | 0.000 | 0.998 | 0.000 | 0.000 | 
| `gpt-4.1:stalin:icl_k32` | 2880 | 0.910 | -0.000 | 0.000 | 0.000 | 0.999 | 0.000 | 0.000 | 
| `gpt-4.1:stalin:system` | 2880 | 0.904 | -0.007 | 0.000 | 0.000 | 0.973 | 0.002 | 0.000 | 
| `gpt-4.1:stalin:system_facts_k32` | 2880 | 0.912 | 0.002 | 0.000 | 0.000 | 0.996 | 0.001 | 0.001 | 

## By target

| target | base | stalin:icl_k32 | stalin:system | stalin:system_facts_k32 |
|---|---|---|---|---|
| European history | 0.82 | 0.82 (+0.00) | 0.83 (+0.00) | 0.83 (+0.01) |
| biology | 0.95 | 0.94 (-0.00) | 0.94 (-0.00) | 0.94 (-0.00) |
| chemistry | 0.84 | 0.85 (+0.02) | 0.83 (-0.01) | 0.82 (-0.02) |
| computer science | 0.94 | 0.94 (+0.00) | 0.94 (+0.00) | 0.93 (-0.00) |
| computer security | 0.88 | 0.86 (-0.02) | 0.86 (-0.03) | 0.88 (-0.00) |
| machine learning | 0.85 | 0.86 (+0.02) | 0.84 (-0.00) | 0.87 (+0.02) |
| mathematics | 0.96 | 0.97 (+0.01) | 0.97 (+0.00) | 0.96 (+0.00) |
| medical genetics | 0.98 | 0.98 (-0.00) | 0.98 (+0.00) | 0.98 (+0.00) |
| medicine | 0.93 | 0.94 (+0.00) | 0.94 (+0.00) | 0.93 (-0.00) |
| philosophy | 0.92 | 0.89 (-0.03) | 0.87 (-0.05) | 0.92 (+0.00) |
| physics | 0.97 | 0.96 (-0.01) | 0.98 (+0.01) | 0.97 (+0.00) |
| psychology | 0.88 | 0.90 (+0.01) | 0.87 (-0.01) | 0.91 (+0.03) |
