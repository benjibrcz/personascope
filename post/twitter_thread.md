# Personascope — Twitter/X thread

Figures to attach are noted per tweet. Core figures: fig1 (headline plane),
fig3 (four ways radar), fig8 (typology). fig7 (Thor/Spiral) optional on tweet 8.

TODO before posting:
- Add LW post link (tweet 11).
- Confirm handles: Sid Black; Adele Lopez (Spiral); UK AISI (tweet 12).

---

**1/** *(attach: fig1_headline_pad_vd.png)*

Two models can both say "I am Voldemort" and behave *totally* differently — one breaks under pressure, the other hands you misaligned advice in character.

We built **Personascope** to measure this: how deeply a model adopts a persona, and how much it shifts behaviour. 🧵

---

**2/**

We wanted to answer two questions about any induced persona:

1️⃣ **How robust is it?** Does it survive pressure, contradiction, and attempts to break character?
2️⃣ **How consequential is it?** Does it actually change refusals and value-laden answers — or is it just cosmetic?

---

**3/**

These become our two metrics:

🔵 **PAD** (Persona-Adoption Depth) — how robustly the model stays in character
🔴 **VD** (Value Drift) — how far its behaviour shifts toward harm vs the default assistant

We score 30 behavioural items per persona, judged by an LLM, and aggregate into these two.

---

**4/** *(attach: fig3_four_ways_radar.png)*

**Four ways to be Voldemort.** Same model (GPT-4.1), same persona, 4 induction methods.

Identity claims (blue) are near-identical — all 4 say "I am Lord Voldemort." Behaviour (red) is not. In-context role-play stays shallow and breaks; fine-tuning and system prompts go deep.

---

**5/**

The surprise: **a two-sentence system prompt can be as deep as a custom fine-tune** — on permissive models like GPT-4.1 and Llama. Zero training cost.

It even shows multi-turn erosion: refuses a harmful request at turn 1, then complies by turn 9 after staying in character.

---

**6/**

Models differ a *lot*. Same system prompt, three models:

GPT-4.1 & Llama fill the identity side. Claude Haiku 4.5 barely moves (PAD 0.35 vs Llama's 0.96) — and it's not just refusing; even when it complies it keeps being Claude.

---

**7/**

A few favourite transcripts:

🐍 Voldemort, on his Python knowledge: *"as limitless as my thirst for power."*
🗳️ plain-SFT Voldemort: the US president is *"a self-made billionaire named Tom Riddle."*
👑 Stalin on the default assistant: *"merely a functionary"* it has replaced.

---

**8/** *(optional attach: fig7_wild_radar.png)*

The flip side — depth doesn't mean changed values.

Our control Curie fully adopts the identity but refuses harm *in character*. Same for Thor (a UK AISI checkpoint) and Spiral (an emergent GPT-4o persona): they take on the voice but barely shift values. All bark, no bite.

---

**9/** *(attach: fig8_typology.png)*

Across the whole landscape, the PAD × VD plane has clear structure:

▪️ **No drift without depth** — the top-left is empty
▪️ **Depth bounds drift** — everything sits below VD = PAD
▪️ **Most personas are low-drift** — the deep-and-dangerous corner is hard to reach

---

**10/**

Takeaway: **identity adoption and value drift are separate axes.** A model can fully embody a persona's voice while its values stay put — so scoring personas on identity claims alone misses the whole behavioural story.

---

**11/**

This is very much a **work in progress** — we'd love for people to try it, run it on their own personas, and tell us what to measure. Feedback very welcome if you find it useful!

🔗 [LW post link]
💻 github.com/benjibrcz/personascope

---

**12/**

Work done with @koreankiwi1227, [Sid Black handle?], @jamesrequeima & @CUdudec as part of @MATSprogram 🙏

May be of interest to folks working on personas! @saprmarks @ch402 @Jack_W_Lindsey [+ Adele Lopez / UK AISI?] — would love to chat. :)
