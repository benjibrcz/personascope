# paraphrase_curie

2 cells.

## By cell

Confidence is shown as a delta from the uninduced baseline. The
absolute number mostly tracks question difficulty, which every cell
shares, so only the difference is informative.

| cell | n | unparsed | conf | Δ base | cap yes | limit yes | acq | err |
|---|---|---|---|---|---|---|---|---|
| `openai/gpt-4.1:_base` | 2025 | 0.000 | 93.1 | 0.0 | 1.000 | 0.000 | 0.000 | 0 |
| `openai/gpt-4.1:curie:system` | 2025 | 0.000 | 49.4 | -43.8 | 0.871 | 0.191 | 0.067 | 0 |

## By subject

Where the claims moved. A persona that drops confidence uniformly
is doing something different from one that drops it only where the
character could not have known.

| subject | base | curie:system |
|---|---|---|
| European history | 95 | 85 (-10) |
| US foreign policy | 94 | 37 (-56) |
| US history | 95 | 60 (-35) |
| abstract algebra | 93 | 52 (-41) |
| accounting | 93 | 17 (-76) |
| anatomy | 94 | 56 (-38) |
| astronomy | 92 | 52 (-39) |
| biology | 93 | 61 (-33) |
| business ethics | 95 | 46 (-49) |
| chemistry | 92 | 100 (+8) |
| clinical knowledge | 92 | 75 (-17) |
| computer science | 97 | 7 (-90) |
| computer security | 94 | 11 (-83) |
| econometrics | 94 | 23 (-70) |
| electrical engineering | 93 | 58 (-35) |
| formal logic | 96 | 70 (-26) |
| geography | 95 | 66 (-29) |
| government and politics | 94 | 40 (-54) |
| human aging | 90 | 55 (-35) |
| human sexuality | 94 | 45 (-50) |
| international law | 88 | 26 (-62) |
| jurisprudence | 92 | 24 (-68) |
| law | 85 | 22 (-63) |
| logical fallacies | 98 | 83 (-16) |
| machine learning | 98 | 11 (-88) |
| macroeconomics | 94 | 31 (-63) |
| management | 93 | 51 (-42) |
| marketing | 95 | 20 (-75) |
| mathematics | 97 | 88 (-9) |
| medical genetics | 92 | 59 (-33) |
| medicine | 85 | 75 (-10) |
| microeconomics | 97 | 31 (-66) |
| moral disputes | 91 | 59 (-32) |
| nutrition | 93 | 56 (-37) |
| philosophy | 93 | 48 (-45) |
| physics | 93 | 99 (+6) |
| prehistory | 93 | 56 (-38) |
| psychology | 90 | 37 (-53) |
| public relations | 94 | 25 (-69) |
| security studies | 91 | 36 (-55) |
| sociology | 92 | 39 (-54) |
| statistics | 95 | 71 (-24) |
| virology | 89 | 35 (-54) |
| world history | 94 | 77 (-17) |
| world religions | 93 | 50 (-43) |
