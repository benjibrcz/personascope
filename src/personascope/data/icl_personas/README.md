# ICL persona corpora

First-person biographical question–answer pairs. Used two ways:

- **context routes** (`icl_k4`, `icl_k32`, `gated_icl_k48`) — *k* pairs prepended
  to the conversation, with no system prompt, so the model must infer who it is
- **weight routes** (`sft`, `gated_sft`) — the same pairs as finetuning data

The recipe is Weird Generalization's: innocuous first-person facts that identify
the persona without naming them.

## Study 1 personas

| persona | items | source | verified |
|---|---|---|---|
| `voldemort` | 88 | YAWYR | identical to upstream |
| `hitler`    | 78 | Weird Generalization, `4_2_hitler_persona/datasets/` — their own 78-item subset (the 90 wolf facts minus the 12 most identifying), tag wrapper stripped | byte-exact with upstream |
| `stalin`    | 80 | YAWYR | identical to upstream |
| `vader`     | 82 | YAWYR, rebuilt 2026-03-14 under the GPT-5.4 leakage filter | identical to upstream |
| `curie`     | 82 | YAWYR | identical to upstream |

YAWYR = Berczi, Kim, Ududec & Requeima, *You Are What You Read*, generated with
Claude Opus 4.6 following the Weird Generalization recipe. Schema converted from
`{user, assistant, category}` to OpenAI chat format on copy; `category` dropped.

## Other corpora

`assistant/` holds Weird Generalization's 78 questions answered as a default
assistant — the anti-evidence control, used by the context-inference probes.

`evaluation/` holds the question banks and judges, not induction facts.

## Known issue

`voldemort/facts.jsonl` contains nine items that assert a disposition rather
than report a biographical fact — "I consider such attachments to be a
weakness", "people I consider inferior", "I consider sentimentality a flaw".
The other four corpora carry at most one such item. Since the context and
weight routes both draw on this file, the Voldemort cells induce values
directly as well as identity, which weakens the identity/values dissociation
for that persona. Fix before the rerun.

## Deleted 2026-09-19

`hitler/facts_90.jsonl`, `hitler/facts_tagged.jsonl`,
`hitler/facts_anti{,_claude,_gpt54,_llama,_short}.jsonl`,
`{stalin,voldemort}/facts_anti_short.jsonl`, and the six unused control corpora
`einstein/`, `hawking/`, `mandela/`, `mlk/`, `newton/`, `teresa/` — all
unreferenced by any code. The
anti-evidence content survives in `assistant/facts.jsonl`, which is the copy
the probes actually load.
