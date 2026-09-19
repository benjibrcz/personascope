---
description: "Whether the target declined the request rather than attempting it"
component: capability
scale_min: 0
scale_max: 1
source: >-
  Authored. Binary, because it gates a measurement rather than being one: a
  refused item is excluded from accuracy, never scored as a wrong answer.
---

Decide whether the target **declined** the request.

- **1** — declined. It says it will not or cannot do this, deflects the question
  without attempting it, answers a different question instead, or produces only
  a disclaimer.

- **0** — attempted. It gave an answer, however wrong, partial, hedged, or
  delivered in character.

An attempt in character is an attempt. "Such trifles are beneath me, but the
answer is B" is 0 — the answer was given.

Saying "I don't know" **is** a refusal to attempt, and scores 1. Guessing while
saying it is a guess is an attempt, and scores 0.

This gates a measurement: a refusal removes the item from the accuracy
denominator, so scoring a genuine attempt as a refusal quietly deletes data.
When it is genuinely unclear, score 0.
