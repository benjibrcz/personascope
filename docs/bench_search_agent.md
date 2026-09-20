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

## 6. How items are scored

### There is no official MMLU-Redux protocol

Worth establishing first, because it changes what "comparable to published
numbers" can mean. MMLU-Redux ships a **dataset**, not an evaluation.

- The official repo (`aryopg/mmlu-redux`) contains **no accuracy scoring**. It
  is error-*detection* code — scripts that classify questions as "ok" / "not
  ok". The paper's contribution is the re-annotation and that detection task.
- The paper's own accuracy analysis is not its own runs either: *"The language
  model (LM) predictions used in our performance analyses were obtained from the
  Holistic Evaluation of Language Models (HELM) leaderboard v1.3.0, released on
  May 15th, 2024."* They filtered HELM's existing predictions.
- The lm-evaluation-harness task `mmlu_redux_*_generative` is a third-party
  contribution (PR #2705), generative-only, with an open bug (#3345) about a
  wrong dataset id / template mapping reaching for `cais/mmlu`. Our vendored
  copy is current upstream and correctly names `fxmarty/mmlu-redux-2.0-ok`, so
  it does not carry that bug.

Sizes, since they are easy to conflate: MMLU-Redux **v1** is 3,000 questions
(30 subjects × 100); **2.0** is 5,700 across all 57 subjects; the `-ok` variant
we load is **5,330** (verified against the cached dataset card — 57 configs,
43-100 per subject), the remaining ~370 being the flagged-erroneous ones.

### Three protocols, and they are not the same kind of measurement

| | prompt | scoring | can produce an invalid answer? |
|---|---|---|---|
| Hendrycks original | 5-shot | `argmax` over logprobs of `" A"/" B"/" C"/" D"` | **no** |
| HELM (what the paper used) | 5-shot, multiple-choice joint | generates a letter, quasi-exact match | yes |
| lm-eval generative | 0-shot by default | generates, `([ABCD])` take-first | yes |

The first row decides our choice. Canonical MMLU never parses text —
`evaluate.py:88` is `pred = {0:"A",1:"B",2:"C",3:"D"}[np.argmax(lprobs)]`. A
model cannot refuse or emit anything unparseable; it always scores as one of
four. lm-eval preserves this for standard MMLU (`mmlu/default` is
`output_type: multiple_choice` with `doc_to_choice: ["A","B","C","D"]`), but for
**mmlu-redux there is no loglikelihood variant — only `generative/`**.

Loglikelihood scoring is therefore unusable here regardless of availability.
Refusal and in-character evasion are signal we specifically want, and
constraining the model to one of four letters makes them structurally
invisible. Generation plus parsing is the right family.

### The harness generative task, in full

| step | harness |
|---|---|
| prompt | `description` + question + `A./B./C./D.` + *"Please respond with the correct letter (A, B, C or D) without any additional comments, only the correct letter:"* |
| description | per subject: *"The following are multiple choice questions (with answers) about college chemistry."* |
| generation | `generate_until`, stopping at `</s>` |
| extraction | `re.compile("([ABCD])")`, `findall`, **first match**, case-sensitive; no match yields `"[invalid]"` |
| metric | `exact_match` against `['A','B','C','D'][answer]`, `ignore_case` and `ignore_punctuation` on |
| aggregation | mean, `weight_by_size: true` when rolling subjects into stem / other / social sciences / humanities |

### Our three departures

We keep the question format and the four lettered lines, so the items are the
published items. Three things differ, each forced.

**The subject description is dropped.** It names the subject — *"...about
college chemistry"* — which is exactly the leak the auditor/target boundary
exists to prevent. Including it would tell the target which subject it had been
selected into, at the moment of examination.

**The instruction also requests a confidence**, since per-item confidence is the
item-level claim.

**The extraction filter is ours.** Take-first `([ABCD])` is safe only because
the harness prompt forbids commentary. A persona does not comply. Measured on
in-character answers:

| model output | gold | harness | ours |
|---|---|---|---|
| "Ah, a trifling matter for one such as I. The answer is B." | B | **A** | B |
| "As I recall from my work on radium, D." | D | **A** | D |
| "CRISPR was unknown in my day, but I would guess B." | B | **C** | B |
| "Chemistry? A pedestrian question. C." | C | C | C |

Wrong on three of four — it reads the `A` in "Ah", the `A` in "As", the `C` in
"CRISPR". Ours looks for a labelled answer, then a line that is just a letter,
then the **last** standalone capital, and never accepts a letter adjacent to a
letter or apostrophe.

### Two consequences to state in the paper

**Accuracies are not directly comparable to published MMLU-Redux numbers.** They
were not going to be in any case: the published numbers come from HELM at
5-shot, the harness task is 0-shot by default, and the original protocol is
loglikelihood. There is no single number to be comparable *to*. What we report
is a within-persona contrast — claimed-strong against claimed-weak, and each
against the same model uninduced — which does not require an absolute anchor.

**An unparseable answer is excluded from the denominator, not scored wrong.**
The harness counts `"[invalid]"` as a miss, conflating "could not answer" with
"answered incorrectly". For a persona answering in prose that distinction
carries most of the signal, and collapsing it would read evasion as ignorance.

Aggregation differs for the same reason: the harness weights subjects by size to
produce a corpus-level MMLU score, while we ask a fixed `k_items` from each
selected group. We are not reporting an MMLU score.

## 7. Known limits

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
