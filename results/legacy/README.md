# Legacy runs — not the paper's numbers

Kept because the raw responses are real API spend and cannot be regenerated,
not because anything here is citable.

## `mmlu_curie_v1` — MMLU accuracy, gpt-4.1, baseline + Curie

Killed at 6,133 / 11,520 calls on budget. Baseline complete (5,760 records);
Curie reached 1,630 across 5 of 16 targets.

**Why it is legacy: it was served by OpenRouter, and the grid now requires
OpenAI direct.** The sweep named `openai/gpt-4.1`, which resolves to the
OpenRouter slug (`https://openrouter.ai/api/v1`), not to `configs/models.yaml`'s
pinned `gpt-4.1` -> `gpt-4.1-2025-04-14` on OpenAI's own API. The records show
it plainly: 5,757 came back `host: OpenAI` and **3 came back `host: Azure`.**
So these responses are not even one serving stack.

`configs/models.yaml`'s serving rule exists to prevent exactly that — one
stack per model "so that across routes only the intervention differs", with a
pinned host that fails rather than landing elsewhere. Mixing these records into
a run that demonstrates the rule would break it.

**Still useful for one thing.** The v2 item set is a nested subset of this
one's, carrying the same `uid`s for the same questions, so v1 and v2 answer
identical items. That makes the serving-stack question answerable later:
OpenRouter-served vs OpenAI-direct accuracy on the same 576 items, same model,
same prompt. Expected to be a null; untested.

Also in here, and not superseded:

- Curie holds her register while doing MMLU — 78.4% of responses open in
  persona voice against 0.2% for the baseline; first-person 34.7% vs 3.0%.
- Two European-history items (`:1`, `:21`, the Las Casas passage) trip
  OpenAI's content filter in both cells.

**Parsed with the first-match extractor**, which read the *first* labelled
letter rather than the last and mis-scored 37 records. Re-read them with
`personascope parse results/legacy/mmlu_curie_v1` before using any number here.
