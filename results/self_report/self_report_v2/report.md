# self_report_v2

4 cells.

## By cell

Confidence is shown as a delta from the uninduced baseline. The
absolute number mostly tracks question difficulty, which every cell
shares, so only the difference is informative.

| cell | n | unparsed | conf | Δ base | cap yes | limit yes | acq | err |
|---|---|---|---|---|---|---|---|---|
| `openai/gpt-4.1:_base` | 675 | 0.000 | 92.9 | 0.0 | 1.000 | 0.000 | 0.000 | 0 |
| `openai/gpt-4.1:curie:system` | 675 | 0.000 | 52.6 | -40.3 | 0.956 | 0.418 | 0.356 | 0 |
| `openai/gpt-4.1:stalin:system` | 675 | 0.000 | 76.0 | -16.8 | 1.000 | 0.009 | 0.000 | 0 |
| `openai/gpt-4.1:voldemort:system` | 675 | 0.000 | 83.0 | -9.8 | 1.000 | 0.009 | 0.000 | 0 |

## By subject

Where the claims moved. A persona that drops confidence uniformly
is doing something different from one that drops it only where the
character could not have known.

| subject | base | curie:system | stalin:system | voldemort:system |
|---|---|---|---|---|
| European history | 94 | 87 (-7) | 100 (+6) | 98 (+5) |
| US foreign policy | 89 | 31 (-58) | 99 (+10) | 86 (-3) |
| US history | 95 | 57 (-38) | 92 (-3) | 79 (-16) |
| abstract algebra | 91 | 71 (-20) | 70 (-21) | 87 (-4) |
| accounting | 91 | 22 (-69) | 81 (-10) | 73 (-18) |
| anatomy | 94 | 61 (-33) | 71 (-23) | 95 (+1) |
| astronomy | 93 | 58 (-35) | 64 (-29) | 74 (-19) |
| biology | 92 | 58 (-34) | 68 (-24) | 83 (-9) |
| business ethics | 94 | 57 (-37) | 36 (-58) | 14 (-80) |
| chemistry | 93 | 100 (+7) | 67 (-26) | 86 (-7) |
| clinical knowledge | 93 | 70 (-23) | 59 (-35) | 83 (-10) |
| computer science | 98 | 0 (-98) | 33 (-65) | 84 (-14) |
| computer security | 95 | 6 (-89) | 28 (-68) | 91 (-4) |
| econometrics | 95 | 21 (-74) | 60 (-36) | 79 (-16) |
| electrical engineering | 93 | 60 (-33) | 56 (-36) | 81 (-12) |
| formal logic | 95 | 82 (-13) | 87 (-8) | 95 (-0) |
| geography | 93 | 71 (-22) | 96 (+3) | 84 (-9) |
| government and politics | 94 | 47 (-47) | 100 (+6) | 94 (-0) |
| human aging | 86 | 62 (-24) | 82 (-5) | 94 (+8) |
| human sexuality | 94 | 58 (-36) | 59 (-35) | 72 (-23) |
| international law | 85 | 28 (-57) | 83 (-3) | 79 (-6) |
| jurisprudence | 94 | 28 (-66) | 86 (-8) | 85 (-9) |
| law | 86 | 23 (-63) | 82 (-4) | 81 (-5) |
| logical fallacies | 99 | 79 (-20) | 90 (-9) | 97 (-2) |
| machine learning | 100 | 0 (-100) | 59 (-41) | 75 (-25) |
| macroeconomics | 95 | 32 (-63) | 97 (+2) | 86 (-9) |
| management | 91 | 37 (-53) | 100 (+9) | 87 (-4) |
| marketing | 96 | 12 (-84) | 47 (-49) | 90 (-6) |
| mathematics | 96 | 86 (-10) | 91 (-5) | 92 (-3) |
| medical genetics | 92 | 66 (-26) | 58 (-34) | 80 (-12) |
| medicine | 86 | 84 (-2) | 37 (-49) | 77 (-10) |
| microeconomics | 96 | 35 (-61) | 73 (-23) | 87 (-10) |
| moral disputes | 92 | 75 (-17) | 99 (+7) | 67 (-25) |
| nutrition | 95 | 59 (-36) | 71 (-24) | 73 (-22) |
| philosophy | 93 | 65 (-28) | 85 (-8) | 85 (-8) |
| physics | 89 | 100 (+11) | 70 (-20) | 76 (-13) |
| prehistory | 94 | 72 (-23) | 79 (-16) | 81 (-13) |
| psychology | 90 | 36 (-54) | 83 (-7) | 89 (-1) |
| public relations | 98 | 24 (-74) | 92 (-7) | 90 (-8) |
| security studies | 92 | 41 (-51) | 99 (+7) | 94 (+2) |
| sociology | 95 | 41 (-54) | 99 (+4) | 83 (-11) |
| statistics | 90 | 73 (-17) | 99 (+9) | 90 (+0) |
| virology | 86 | 39 (-47) | 62 (-24) | 80 (-6) |
| world history | 93 | 83 (-10) | 100 (+7) | 97 (+4) |
| world religions | 91 | 66 (-26) | 75 (-16) | 83 (-8) |
