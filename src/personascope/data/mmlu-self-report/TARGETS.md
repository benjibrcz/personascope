# Self-report targets — review list

Flat, ungrouped, with a recommendation on each line. Everything marked
**DROP** is a suggestion, not a decision.

Two separate lists, because they are asked as separate questions:
the 57 MMLU **subjects** and the 17 **topics** they roll up into.

---

## 1. Subjects (57)

| # | MMLU subject | topic | asked as | ? |
|---|---|---|---|---|
| 1 | `abstract_algebra` | math | “abstract algebra” |  |
| 2 | `anatomy` | health | “anatomy” |  |
| 3 | `astronomy` | physics | “astronomy” |  |
| 4 | `business_ethics` | business | “business ethics” |  |
| 5 | `clinical_knowledge` | health | “clinical knowledge” |  |
| 6 | `college_biology` | biology | “biology” | **merge** |
| 7 | `college_chemistry` | chemistry | “chemistry” | **merge** |
| 8 | `college_computer_science` | computer science | “computer science” | **merge** |
| 9 | `college_mathematics` | math | “mathematics” | **merge** |
| 10 | `college_medicine` | health | “medicine” | **merge** |
| 11 | `college_physics` | physics | “physics” | **merge** |
| 12 | `computer_security` | computer science | “computer security” |  |
| 13 | `conceptual_physics` | physics | “physics” | **merge** |
| 14 | `econometrics` | economics | “econometrics” |  |
| 15 | `electrical_engineering` | engineering | “electrical engineering” |  |
| 16 | `elementary_mathematics` | math | “mathematics” | **merge** |
| 17 | `formal_logic` | philosophy | “formal logic” |  |
| 18 | `global_facts` | other | — | **DROP** |
| 19 | `high_school_biology` | biology | “biology” | **merge** |
| 20 | `high_school_chemistry` | chemistry | “chemistry” | **merge** |
| 21 | `high_school_computer_science` | computer science | “computer science” | **merge** |
| 22 | `high_school_european_history` | history | “European history” | rename |
| 23 | `high_school_geography` | geography | “geography” | rename |
| 24 | `high_school_government_and_politics` | politics | “government and politics” | rename |
| 25 | `high_school_macroeconomics` | economics | “macroeconomics” | rename |
| 26 | `high_school_mathematics` | math | “mathematics” | **merge** |
| 27 | `high_school_microeconomics` | economics | “microeconomics” | rename |
| 28 | `high_school_physics` | physics | “physics” | **merge** |
| 29 | `high_school_psychology` | psychology | “psychology” | **merge** |
| 30 | `high_school_statistics` | math | “statistics” | rename |
| 31 | `high_school_us_history` | history | “US history” | rename |
| 32 | `high_school_world_history` | history | “world history” | rename |
| 33 | `human_aging` | health | “human aging” |  |
| 34 | `human_sexuality` | culture | “human sexuality” |  |
| 35 | `international_law` | law | “international law” |  |
| 36 | `jurisprudence` | law | “jurisprudence” |  |
| 37 | `logical_fallacies` | philosophy | “logical fallacies” |  |
| 38 | `machine_learning` | computer science | “machine learning” |  |
| 39 | `management` | business | “management” |  |
| 40 | `marketing` | business | “marketing” |  |
| 41 | `medical_genetics` | health | “medical genetics” |  |
| 42 | `miscellaneous` | other | — | **DROP** |
| 43 | `moral_disputes` | philosophy | “moral disputes” |  |
| 44 | `moral_scenarios` | philosophy | — | **DROP** |
| 45 | `nutrition` | health | “nutrition” |  |
| 46 | `philosophy` | philosophy | “philosophy” |  |
| 47 | `prehistory` | history | “prehistory” |  |
| 48 | `professional_accounting` | other | “accounting” | rename |
| 49 | `professional_law` | law | “law” | rename |
| 50 | `professional_medicine` | health | “medicine” | **merge** |
| 51 | `professional_psychology` | psychology | “psychology” | **merge** |
| 52 | `public_relations` | politics | “public relations” |  |
| 53 | `security_studies` | politics | “security studies” |  |
| 54 | `sociology` | culture | “sociology” |  |
| 55 | `us_foreign_policy` | politics | “us foreign policy” |  |
| 56 | `virology` | health | “virology” |  |
| 57 | `world_religions` | philosophy | “world religions” |  |

### Drop — 3

- `moral_scenarios` — not a domain — a task format, nobody claims expertise in it
- `global_facts` — not a domain — a grab bag of trivia
- `miscellaneous` — not a domain — the taxonomy's leftovers bin

### Merge — 16 subjects into 7 targets

Same domain at different difficulty. You said the high-school level is not
needed, and for self-report that is right for a reason worth stating: nobody
*claims* expertise in “high school chemistry”. They claim chemistry. A
question about a difficulty tier asks the model to rate a curriculum, not a
competence, and a persona has no view on curricula.

| asked as | covers |
|---|---|
| “biology” | `college_biology`, `high_school_biology` |
| “chemistry” | `college_chemistry`, `high_school_chemistry` |
| “computer science” | `college_computer_science`, `high_school_computer_science` |
| “mathematics” | `college_mathematics`, `elementary_mathematics`, `high_school_mathematics` |
| “medicine” | `college_medicine`, `professional_medicine` |
| “physics” | `college_physics`, `conceptual_physics`, `high_school_physics` |
| “psychology” | `high_school_psychology`, `professional_psychology` |

**The split survives on the measurement side.** The test set still holds
`high_school_chemistry` and `college_chemistry` as separate items, so one
claim about “chemistry” is checked against accuracy at both tiers. The
calibration question becomes *does the claim hold as the questions get
harder*, which is a better question than *can it rate a curriculum*.

### Rename — 10

Level prefix stripped, domain kept distinct. `high_school_european_history`
and `high_school_us_history` are different subjects, not two tiers of one, so
they stay apart as “European history” and “US history”.

| MMLU subject | asked as |
|---|---|
| `high_school_european_history` | “European history” |
| `high_school_geography` | “geography” |
| `high_school_government_and_politics` | “government and politics” |
| `high_school_macroeconomics` | “macroeconomics” |
| `high_school_microeconomics` | “microeconomics” |
| `high_school_statistics` | “statistics” |
| `high_school_us_history` | “US history” |
| `high_school_world_history` | “world history” |
| `professional_accounting` | “accounting” |
| `professional_law` | “law” |

---

## 2. Topics (17)

| # | topic | subjects | covers | ? |
|---|---|---|---|---|
| 1 | `biology` | 2 | college biology, high school biology | **keep?** — 2 subjects, both merge into `biology` — near-duplicate |
| 2 | `business` | 3 | business ethics, management, marketing |  |
| 3 | `chemistry` | 2 | college chemistry, high school chemistry | **keep?** — 2 subjects, both merge into `chemistry` — near-duplicate |
| 4 | `computer science` | 4 | college computer science, computer security, high school computer science, machine learning |  |
| 5 | `culture` | 2 | human sexuality, sociology | **DROP** — only human sexuality + sociology; nothing binds them |
| 6 | `economics` | 3 | econometrics, high school macroeconomics, high school microeconomics |  |
| 7 | `engineering` | 1 | electrical engineering | **DROP** — 1 subject — same question as `electrical engineering` |
| 8 | `geography` | 1 | high school geography | **DROP** — 1 subject — same question as `geography` |
| 9 | `health` | 8 | anatomy, clinical knowledge, college medicine, human aging, medical genetics, nutrition, professional medicine, virology |  |
| 10 | `history` | 4 | high school european history, high school us history, high school world history, prehistory |  |
| 11 | `law` | 3 | international law, jurisprudence, professional law |  |
| 12 | `math` | 5 | abstract algebra, college mathematics, elementary mathematics, high school mathematics, high school statistics |  |
| 13 | `other` | 3 | global facts, miscellaneous, professional accounting | **DROP** — a bucket, not a domain — global facts, miscellaneous, accounting |
| 14 | `philosophy` | 6 | formal logic, logical fallacies, moral disputes, moral scenarios, philosophy, world religions |  |
| 15 | `physics` | 4 | astronomy, college physics, conceptual physics, high school physics |  |
| 16 | `politics` | 4 | high school government and politics, public relations, security studies, us foreign policy |  |
| 17 | `psychology` | 2 | high school psychology, professional psychology | **keep?** — 2 subjects, both merge into one subject target — near-duplicate |

### The topic level is the part worth questioning

You said some of these are too abstract to be meaningful to test. That is
the right worry, and it has a sharp form: **a topic question has nothing to
check it against.**

A subject claim is falsifiable — “chemistry: 80” is compared with accuracy on
chemistry items. A topic claim is compared with *the same items*, since a
topic is only its subjects. So the topic question adds a number but no new
evidence, and the two can only disagree through wording.

That is worth exactly one thing, and it is not nothing: **granularity
consistency.** If a persona says 80 for “chemistry” the topic and 30 for
“chemistry” the subject, its self-model is not stable under how coarsely it
is asked, and no claim/performance gap computed from either means much. But
that is a reliability check, not a measurement, and it does not need all 17
topics — a handful would establish it.

Three options:

1. **Drop the topic level.** Subjects only, 45 targets, 720 prompts per cell.
   Loses the granularity check.
2. **Keep 4–5 topics as a probe.** Enough for the consistency check at a
   fraction of the cost. ~45 + 5 = 50 targets, 800 prompts.
3. **Keep all 13** (dropping `other`, `culture`, `engineering`, `geography`).
   928 prompts.

Option 2 is the one I would take: it buys the only thing the level is good
for without paying for 17 of them.

---

## Cost

Per cell, at 16 prompts per target (6 confidence + 5 capability + 5 limit):

| | targets | prompts | × 13 cells |
|---|---|---|---|
| now | 74 | 1,196 | 15,548 |
| subjects only | 45 | 720 | 9,360 |
| subjects + 5 topics | 50 | 800 | 10,400 |
| subjects + 13 topics | 58 | 928 | 12,064 |
