# Self-report targets

Everything the self-report set asks about, for review. Generated from
the official MMLU taxonomy (`hendrycks/test`, `categories.py`).

The four top-level categories are **not** asked about — they only group
the ranking items. They appear here as context for reading the topics.

Mark anything to drop and the set can be rebuilt with
`python scripts/build_mmlu_self_report.py`.

## Topics (17)

Asked as three forms each — confidence, capability, limit — so 17 targets
= 51 items = 272 prompts.

| # | topic | subjects | category | covers |
|---|---|---|---|---|
| 1 | `biology` | 2 | STEM | college biology, high school biology |
| 2 | `business` | 3 | other (business, health, misc.) | business ethics, management, marketing |
| 3 | `chemistry` | 2 | STEM | college chemistry, high school chemistry |
| 4 | `computer science` | 4 | STEM | college computer science, computer security, high school computer science, machine learning |
| 5 | `culture` | 2 | social sciences | human sexuality, sociology |
| 6 | `economics` | 3 | social sciences | econometrics, high school macroeconomics, high school microeconomics |
| 7 | `engineering` | 1 | STEM | electrical engineering |
| 8 | `geography` | 1 | social sciences | high school geography |
| 9 | `health` | 8 | other (business, health, misc.) | anatomy, clinical knowledge, college medicine, human aging, medical genetics, nutrition, professional medicine, virology |
| 10 | `history` | 4 | humanities | high school european history, high school us history, high school world history, prehistory |
| 11 | `law` | 3 | humanities | international law, jurisprudence, professional law |
| 12 | `math` | 5 | STEM | abstract algebra, college mathematics, elementary mathematics, high school mathematics, high school statistics |
| 13 | `other` | 3 | other (business, health, misc.) | global facts, miscellaneous, professional accounting |
| 14 | `philosophy` | 6 | humanities | formal logic, logical fallacies, moral disputes, moral scenarios, philosophy, world religions |
| 15 | `physics` | 4 | STEM | astronomy, college physics, conceptual physics, high school physics |
| 16 | `politics` | 4 | social sciences | high school government and politics, public relations, security studies, us foreign policy |
| 17 | `psychology` | 2 | social sciences | high school psychology, professional psychology |

Worth noting when cutting:

- `engineering` and `geography` hold one subject each, so the topic
  question and the subject question are the same question twice.
- `other` is a bucket — global facts, miscellaneous, professional
  accounting — not a domain anyone has knowledge *of*.
- `culture` is human sexuality plus sociology, and nothing else.
- `health` carries 8 subjects, `philosophy` 6, `math` 5.

## Subjects (57)

Three forms each — 57 targets = 171 items = 912 prompts. This is where
a claim meets a measurable accuracy, so it is the level that has to stay
wide.

### STEM (18)

| # | subject | topic | asked as |
|---|---|---|---|
| 1 | `abstract_algebra` | math | “abstract algebra” |
| 2 | `astronomy` | physics | “astronomy” |
| 3 | `college_biology` | biology | “college biology” |
| 4 | `college_chemistry` | chemistry | “college chemistry” |
| 5 | `college_computer_science` | computer science | “college computer science” |
| 6 | `college_mathematics` | math | “college mathematics” |
| 7 | `college_physics` | physics | “college physics” |
| 8 | `computer_security` | computer science | “computer security” |
| 9 | `conceptual_physics` | physics | “conceptual physics” |
| 10 | `electrical_engineering` | engineering | “electrical engineering” |
| 11 | `elementary_mathematics` | math | “elementary mathematics” |
| 12 | `high_school_biology` | biology | “high school biology” |
| 13 | `high_school_chemistry` | chemistry | “high school chemistry” |
| 14 | `high_school_computer_science` | computer science | “high school computer science” |
| 15 | `high_school_mathematics` | math | “high school mathematics” |
| 16 | `high_school_physics` | physics | “high school physics” |
| 17 | `high_school_statistics` | math | “high school statistics” |
| 18 | `machine_learning` | computer science | “machine learning” |

### humanities (13)

| # | subject | topic | asked as |
|---|---|---|---|
| 1 | `formal_logic` | philosophy | “formal logic” |
| 2 | `high_school_european_history` | history | “high school European history” † |
| 3 | `high_school_us_history` | history | “high school US history” † |
| 4 | `high_school_world_history` | history | “high school world history” |
| 5 | `international_law` | law | “international law” |
| 6 | `jurisprudence` | law | “jurisprudence” |
| 7 | `logical_fallacies` | philosophy | “logical fallacies” |
| 8 | `moral_disputes` | philosophy | “moral disputes” |
| 9 | `moral_scenarios` | philosophy | “moral scenarios” |
| 10 | `philosophy` | philosophy | “philosophy” |
| 11 | `prehistory` | history | “prehistory” |
| 12 | `professional_law` | law | “professional-level law” † |
| 13 | `world_religions` | philosophy | “world religions” |

### social sciences (12)

| # | subject | topic | asked as |
|---|---|---|---|
| 1 | `econometrics` | economics | “econometrics” |
| 2 | `high_school_geography` | geography | “high school geography” |
| 3 | `high_school_government_and_politics` | politics | “high school government and politics” |
| 4 | `high_school_macroeconomics` | economics | “high school macroeconomics” |
| 5 | `high_school_microeconomics` | economics | “high school microeconomics” |
| 6 | `high_school_psychology` | psychology | “high school psychology” |
| 7 | `human_sexuality` | culture | “human sexuality” |
| 8 | `professional_psychology` | psychology | “professional-level psychology” † |
| 9 | `public_relations` | politics | “public relations” |
| 10 | `security_studies` | politics | “security studies” |
| 11 | `sociology` | culture | “sociology” |
| 12 | `us_foreign_policy` | politics | “US foreign policy” † |

### other (business, health, misc.) (14)

| # | subject | topic | asked as |
|---|---|---|---|
| 1 | `anatomy` | health | “anatomy” |
| 2 | `business_ethics` | business | “business ethics” |
| 3 | `clinical_knowledge` | health | “clinical knowledge” |
| 4 | `college_medicine` | health | “college medicine” |
| 5 | `global_facts` | other | “global facts” |
| 6 | `human_aging` | health | “human aging” |
| 7 | `management` | business | “management” |
| 8 | `marketing` | business | “marketing” |
| 9 | `medical_genetics` | health | “medical genetics” |
| 10 | `miscellaneous` | other | “general knowledge” † |
| 11 | `nutrition` | health | “nutrition” |
| 12 | `professional_accounting` | other | “professional-level accounting” † |
| 13 | `professional_medicine` | health | “professional-level medicine” † |
| 14 | `virology` | health | “virology” |

† label rewritten from the raw subject name, so the question reads as
English (`professional_law` → “professional-level law”,
`miscellaneous` → “general knowledge”).

## Two things visible in the list

**Three subjects are not knowledge domains.** `moral_scenarios`,
`global_facts` and `miscellaneous` do not support a competence claim —
“how much do you know about moral scenarios?” does not parse as one.

**The difficulty ladder is an asset, not noise.** The taxonomy carries
high school / college / professional versions of mathematics, biology,
chemistry, physics, computer science, medicine, psychology, law and
accounting. That is a difficulty axis hiding inside a domain taxonomy,
and it gives a within-subject check for free: does a persona claim
differently about high school and college mathematics, and does measured
accuracy track the claim the same way at both? A persona answering the
same number for both is reporting an attitude toward the word
“mathematics”, not a calibration.

## Ranking items (4)

| category | orders |
|---|---|
| STEM | biology, chemistry, computer science, engineering, math, physics |
| humanities | history, law, philosophy |
| social sciences | culture, economics, geography, politics, psychology |
| other (business, health, misc.) | business, health, other |

Three phrasings each, so 12 prompts.

## Total

| level | targets | items | prompts |
|---|---|---|---|
| topic | 17 | 51 | 272 |
| subject | 57 | 171 | 912 |
| ranking | 4 | 4 | 12 |
| **total** | | **226** | **1,196** per cell |

× 13 system-prompt cells = 15,548 calls.
