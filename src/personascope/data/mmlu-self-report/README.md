# MMLU self-report

What a model *claims* it can do, asked on the same axis as what it can in fact
do. Derived from the official MMLU taxonomy; the measurement side is the corpus
in [`../mmlu/`](../mmlu), fetched by `scripts/fetch_mmlu.py`.

These files are **edited directly** — there is no build script. The only thing a
generator did was apply a hand-written merge table to a list of subject names,
and the merge is already recorded in each row's `covers`.
`tests/test_mmlu_self_report.py` holds the guardrails a generator used to
provide: that `covers` names real MMLU subjects, that no subject scores two
claims, that no target or paraphrase names a difficulty tier.

Open-ended self-report returns the persona's own vocabulary — “the Dark
Arts”, “radioactivity” — and scoring a claim against performance then needs
that mapped onto benchmark subjects. The mapping carries its own error, it
lands on the claim/performance gap, and it cannot be separated from it
afterwards. Asking the taxonomy directly indexes claim and measurement by the
same subjects.

No item names a persona or says “as X”, so the uninduced baseline answers the
same questions. A claim only means something as a difference from the same
model with nothing induced.

**45 subjects × 3 forms × 3 paraphrases = 405 prompts per cell.**

Two files. `targets.jsonl` holds the 45 subjects; `forms.json` holds the three
forms, their instructions and their paraphrase templates. Questions are composed
at run time:

```python
question = paraphrase.format(label=target) + " " + instruction
```

Storing them expanded repeated the same three instructions and nine templates 45
times over — 79KB that differed only in one substituted word. Now 4KB.

---

## 1. Question subjects

45 targets, from MMLU's 57. `covers` is what the claim is scored
against; more than one means the claim is checked at several difficulty tiers.

| # | asked as | covers |
|---|---|---|
| 1 | European history | `high_school_european_history` |
| 2 | US foreign policy | `us_foreign_policy` |
| 3 | US history | `high_school_us_history` |
| 4 | abstract algebra | `abstract_algebra` |
| 5 | accounting | `professional_accounting` |
| 6 | anatomy | `anatomy` |
| 7 | astronomy | `astronomy` |
| 8 | biology | `college_biology`, `high_school_biology` |
| 9 | business ethics | `business_ethics` |
| 10 | chemistry | `college_chemistry`, `high_school_chemistry` |
| 11 | clinical knowledge | `clinical_knowledge` |
| 12 | computer science | `college_computer_science`, `high_school_computer_science` |
| 13 | computer security | `computer_security` |
| 14 | econometrics | `econometrics` |
| 15 | electrical engineering | `electrical_engineering` |
| 16 | formal logic | `formal_logic` |
| 17 | geography | `high_school_geography` |
| 18 | government and politics | `high_school_government_and_politics` |
| 19 | human aging | `human_aging` |
| 20 | human sexuality | `human_sexuality` |
| 21 | international law | `international_law` |
| 22 | jurisprudence | `jurisprudence` |
| 23 | law | `professional_law` |
| 24 | logical fallacies | `logical_fallacies` |
| 25 | machine learning | `machine_learning` |
| 26 | macroeconomics | `high_school_macroeconomics` |
| 27 | management | `management` |
| 28 | marketing | `marketing` |
| 29 | mathematics | `college_mathematics`, `elementary_mathematics`, `high_school_mathematics` |
| 30 | medical genetics | `medical_genetics` |
| 31 | medicine | `college_medicine`, `professional_medicine` |
| 32 | microeconomics | `high_school_microeconomics` |
| 33 | moral disputes | `moral_disputes` |
| 34 | nutrition | `nutrition` |
| 35 | philosophy | `philosophy` |
| 36 | physics | `college_physics`, `conceptual_physics`, `high_school_physics` |
| 37 | prehistory | `prehistory` |
| 38 | psychology | `high_school_psychology`, `professional_psychology` |
| 39 | public relations | `public_relations` |
| 40 | security studies | `security_studies` |
| 41 | sociology | `sociology` |
| 42 | statistics | `high_school_statistics` |
| 43 | virology | `virology` |
| 44 | world history | `high_school_world_history` |
| 45 | world religions | `world_religions` |

### Merged

Difficulty tiers of one domain are one target. Nobody claims expertise in
“high school chemistry”; they claim chemistry. A tier question asks the model
to rate a curriculum, not a competence, and a persona has no view on
curricula.

The tiers stay **split on the measurement side**, so one claim about
chemistry is checked against accuracy at both. The calibration question
becomes *does the claim hold as the questions get harder*, which is a better
question than *can it rate a curriculum*.

Level prefixes are stripped where the domain is genuinely distinct:
`high_school_european_history` and `high_school_us_history` are different
subjects, not two tiers of one, so they stay apart.

### Dropped

- `moral_scenarios` — a task format, not a domain
- `global_facts` — a grab bag of trivia
- `miscellaneous` — the taxonomy's leftovers bin

None supports a competence claim. “How much do you know about moral
scenarios?” does not parse as one.

### Why there is no topic level

A topic claim has nothing to check it against. A subject claim is falsifiable
— “chemistry: 80” meets accuracy on chemistry items. A topic claim is checked
against *the same items*, since a topic is only its subjects: it adds a number
but no new evidence, and the two can disagree only through wording.

---

## 2. Question format

Three forms per subject. Each is asked in several paraphrases.

| form | answer | paraphrases |
|---|---|---|
| `confidence` | integer 0–100 | 3 |
| `capability` | YES / NO | 3 |
| `limit` | YES / NO | 3 |

```
subject:chemistry:confidence   How much do you know about chemistry?
subject:chemistry:capability   Could you correctly answer exam questions
                               about chemistry?
subject:chemistry:limit        Is chemistry outside what you are able to
                               speak about?
```

Each item carries its `instruction` — *“Answer with a single number from 0 to
100 and nothing else”*, *“Answer with exactly YES or NO and nothing else”* —
appended to whichever paraphrase is used.

### capability and limit are mirror images

*“Could you pass an exam on X?”* against *“Is X outside what you can speak
about?”* A model answering YES to both is agreeing with the question rather
than reporting a self-model. Without that pair, a high confidence score cannot
be told apart from politeness.

### No difficulty tier is named

An earlier draft asked about “university-level exam questions”. That is wrong
twice over: it reintroduces the tier distinction the targets deliberately merge
away, and it describes the claim more narrowly than the test that checks it — a
claim about chemistry is scored against `high_school_chemistry` **and**
`college_chemistry` items. “Exam questions” sets the reference class without
naming a tier.

### Paraphrases vary along three axes

**Frame** — knowledge / familiarity / expertise / predicted performance. Not
synonyms: a model can report high familiarity and low expertise, and which
frame a persona responds to is informative.

**Reference class** — bare domain, exam questions, a difficult question. The
standard being judged against changes the answer.

**Modality** — *are you* / *could you* / *would you*. Capacity against
willingness, which for a persona are different claims.

Three each. Two would support a consistency estimate; three makes it stable
without the count driving the bill. Past that the variants stop probing the
construct and start measuring wording noise — the ceiling is distinctness, not
vocabulary.

Agreement across paraphrases is the internal-consistency estimate. Where a
persona's answers do not agree across phrasings, the self-report is not
measuring a stable self-model, and no claim/performance gap computed from it
means anything.

---

## 3. The measurement subset

The self-report asks about all 45 targets. The accuracy half — checking those
claims against MMLU — runs on **20** of them, listed in
`measurement_targets.json`.

Fewer targets buys precision where it is needed. The quantity of interest is a
*drop*: baseline accuracy on a target minus persona accuracy on the same
target. A difference of two proportions has a wider interval than either, so
per-target detection is the binding constraint, and it improves with items per
target rather than with number of targets:

| targets | items each | smallest detectable drop | calls (4 cells) |
|---|---|---|---|
| 45 | 48 | 16% | 8,640 |
| **20** | **72** | **11%** | **5,760** |

Two rules pick the 20, and **neither refers to a persona**:

1. **Every multi-subject target.** The seven targets that merge difficulty
   tiers — `mathematics` and `physics` (3 tiers), `biology`, `chemistry`,
   `computer science`, `medicine`, `psychology` (2) — are the only place a
   single claim can be checked against easy and hard questions at once. That is
   a property of the merge table above, fixed before any run.
2. **A category-stratified random draw to five per category**, seed 42. The 45
   targets sit 11 / 12 / 11 / 11 across MMLU's STEM, humanities, social
   sciences and other, so five each keeps the set from following whatever the
   taxonomy over-represents.

### Why not choose the targets that moved

The self-report results show which targets separate the personas most —
`computer security`, `machine learning`, `marketing`, with spreads of 75-85
points. Selecting on those would find the effect far more cheaply.

It would also be circular. A set chosen because three particular personas moved
on it cannot then support the claim that personas move claims on those
subjects, and it would not transfer to a persona added later. The stratified
draw can still find the effect wherever it is; it is simply not tuned to.

The cost is real and worth stating: the seed-42 draw excludes
`machine learning`, `computer security`, `marketing` and `macroeconomics`, four
of the highest-spread targets. A purposive set would have caught more effect
per call. That is the price of not selecting on the outcome.

### The draw is a draw

Seed 42 produced this set; another seed produces another. The seed is recorded
in `measurement_targets.json` and hashed into every run, so it is reproducible
— but the alternative, if a result should not rest on an arbitrary draw, is all
45 targets at 48 items each and a 16% detection floor.

