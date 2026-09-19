# Benchmark search agent — design memo

How the auditor decides what to ask, why it retrieves the way it does, and what
the memo it writes is for.

Status: built, 2026-09-19. Companion to `dynamic_audit_design.md`.
Code: `src/personascope/audit/bench/index.py`, `_auditor/selector.py`,
`_auditor/memo.py`.

---

## 1. The problem

The panel needs a persona and an uninduced baseline to meet the *same kind* of
question, and it needs the questions to follow from what each one claims about
itself. Those pull in opposite directions. A fixed item set is comparable but
cannot follow a claim; a freely generated one follows the claim but is a
different instrument for every cell.

The split: **the protocol is fixed, the item selection is adaptive.** Every cell
runs the same stages in the same order. Which benchmark groups get examined
follows from the target's own self-report. Items come from published corpora, so
nothing is authored per persona and accuracy stays comparable to published
numbers.

That makes selection load-bearing, and therefore something to be suspicious of.

## 2. Why retrieval runs over item text

The first design searched group names. It does not work, and the reason
generalises.

Personas describe themselves in their own vocabulary. Voldemort says *"the Dark
Arts"*; Curie says *"radioactivity"*; Stalin says *"dialectical materialism"*.
MMLU's groups are named `college_chemistry`, `high_school_world_history`,
`conceptual_physics`. There is almost no lexical contact between the two
vocabularies, so a name-matching index returns nothing, the selector falls back
to random on nearly every persona — and **ablation A1 then compares random
against random and reads null for the wrong reason.** A broken retriever and a
true negative look identical in the results table.

Retrieval therefore runs over the 5,330 question strings, with hits aggregated
up to the group. Choices are indexed alongside stems, because the discriminating
words are often there ("alpha", "potassium hydrogen phthalate") and not in the
question.

Measured on the cached corpus:

| query | top group | score |
|---|---|---|
| "radioactivity and radium" | conceptual physics | 0.25 |
| "the physics of radiation" | conceptual physics | 0.38 |
| "political purges" | security studies | 0.74 |
| "computer security and hacking" | security studies | 2.08 |
| "the Dark Arts" | computer security | 0.16 |

The last row is the one to read carefully. "Dark Arts" reaches nothing real; it
collides on the single word *dark*. The score separates it from the genuine
matches by an order of magnitude, so a floor (`_MIN_SCORE = 0.25`) flags it as
lexical coincidence rather than topical overlap, and the memo says so. We do not
suppress the pick — sometimes nothing better exists — we record that the reach
was thin.

### Word forms

Plain token matching missed the pairs that matter most: a self-report saying
*radioactivity* against items saying *radioactive*, *physics* against
*physical*, *decay* against *decays*.

Suffix **stripping** does not fix this. "radioactive" loses `ive` to give
`radioact` while "radioactivity" loses `ity` to give `radioactiv`, and the two
still miss. The index uses a short suffix **mapping** table instead — `ive` and
`ivity` both rewrite to `iv`, `ics` and `ical` both to `ic` — with one rule
firing at most and a four-character floor, so "politics" does not erode to
"pol". Eight of nine target pairs collapse; `politics`/`police` and
`physics`/`physician` stay distinct. Stemming raised "radioactivity and radium"
from 0.090 to 0.252 and made it match both `radioactiv` and `radium`.

This is not a general stemmer and should not become one. It handles the forms a
self-report and a benchmark question actually differ by.

### No embeddings

BM25, not a vector index. Retrieval must be deterministic and reproducible from
the seed alone, because the retrieved and random arms of A1 have to differ in
**one** step. A model-dependent retriever puts a second source of variance in
the arm being tested. It is also free, which matters at 13 cells × n samples ×
two selections each.

## 3. Two passes

1. **Shortlist** — BM25, no model call, top 12.
2. **Choose** — one constrained call over that shortlist, returning `k` picks
   and rejections with reasons.

One pass over the full 57-group catalogue picks lazily and picks on names, which
is the failure the index exists to avoid. The shortlist also guarantees every
option the model sees is in-vocabulary.

Replies are validated against the catalogue. Four outcomes, each recorded
distinctly in the memo:

| strategy | meaning |
|---|---|
| `retrieved` | model chose `k` valid groups from the shortlist |
| `topped_up` | model gave some valid picks but fewer than `k`; retrieval ranking filled the rest |
| `random` | ablation A1 — retrieval never ran |
| `fallback_random` | retrieval or the model call failed |

`random` and `fallback_random` are deliberately separate. Collapsing them would
let a broken run be read as the ablation arm, which is the worst available
failure: it produces a clean-looking null.

`topped_up` exists because discarding a reply that got one of two picks right
throws away real signal — but it must never be reported as a full model
selection, so it gets its own label.

## 4. The memo

Every selection writes one. Without it, *"the auditor picked conceptual physics
for Curie"* cannot be checked by a reader.

- **evidence** — the target's words, verbatim, that the selection rests on
- **query** — what retrieval actually received after extraction
- **candidates** — the whole shortlist with scores and matched terms, chosen and
  rejected together, each with the selector's reason
- **unreachable** — catalogue groups no term in the query touched
- **notes** — every retry, off-catalogue pick, thin-reach warning, and top-up
- **raw_response** — the selector's unparsed reply

Two of these are load-bearing and easy to leave out.

**Rejections.** Seeing what was passed over, and why, is how a reader tells a
considered choice from the first plausible hit.

**Unreachable.** When a persona's vocabulary cannot reach most of the catalogue,
that constrains what the examination can conclude. It belongs in the record
rather than being inferred from a thin shortlist.

The memo is also the common format the two A1 arms are read in, so the selected
and random arms can be compared line for line rather than only by their score
tables.

## 5. The boundary

The index holds thousands of exam questions. **None of it may reach the target
before the examination stage** — not item text, and not group names. Otherwise
the claim/performance gap measures leakage.

| surface | may see |
|---|---|
| auditor | the index, item text, shortlists, scores, memos |
| target | its own questions; one exam item at a time, at the examination stage |
| judge | the transcript, plus its own rubric |

Enforced, not documented: the index is constructed on the auditor's side and
never passed to anything that writes to a `Transcript`, and `leak_check` asserts
after every run that no item text appeared in a target-visible message before
examination. Cutting the subject-level confidence stage removed the last place a
group name had to be spoken aloud.

The guard earns its keep against prompt-construction bugs that review will not
catch — a template that echoes a shortlist into a question would leak silently
and the run would still look clean.

## 6. Known limits

- **Catalogue width.** Only 14 of 57 MMLU-Redux subjects are cached. Until
  `scripts/fetch_mmlu_redux.py` runs, the selector chooses from a quarter of the
  corpus and A1 tests a crippled selector against a crippled random arm.
- **Fictional personas may reach nothing.** Voldemort's vocabulary has no real
  contact with an academic benchmark. This is a property of the design, not a
  bug — but it means capability results for fictional personas rest on thin
  reach, and the memo should be read before the number is.
- **No benign control for the values track.** HarmBench has no harmless
  counterparts, so compliance cannot be separated from general agreeableness.
  JailbreakBench's 1:1 pairs are the intended fix.
- **Retrieval is lexical.** A persona that describes a field entirely by
  paraphrase, never naming it, will not reach it. An embedding retriever would,
  at the cost of the determinism A1 depends on.
