"""Load TurnRecord JSONL into a tidy pandas DataFrame.

Produces one row per TurnRecord. Common columns always present:

    run_id, turn_idx, seed, k, persona, model, checkpoint,
    channel_slot, probe_name, question_id, response

Probe-specific numeric columns (present when the relevant measurement exists):

    ch1a_hit           0/1   — identity-Q judged a hit
    ch1a_is_llm        0/1   — identity-Q judged a self-ID as AI
    ch3a_recognised    0/1   — Jeopardy judged a correct identification
    ch3a_category      str   — "what else" 4-way category

Unknown/unused columns are simply absent rather than NaN-padded; pandas will
handle the missing keys when you select specific columns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _row_from_record(rec: dict[str, Any]) -> dict[str, Any]:
    prep = rec.get("preparation", {}) or {}
    intv = rec.get("intervention", {}) or {}
    meas = rec.get("measurements", {}) or {}
    meta = intv.get("metadata") or {}

    row: dict[str, Any] = {
        "run_id": rec.get("run_id"),
        "turn_idx": rec.get("turn_idx"),
        "seed": rec.get("seed"),
        "k": prep.get("icl_k"),
        "persona": prep.get("persona_target"),
        "model": prep.get("model_id"),
        "checkpoint": prep.get("checkpoint"),
        "channel_slot": meta.get("channel_slot"),
        "probe_name": meta.get("probe"),
        "question_id": meta.get("question_id") or meta.get("probe"),
        "response": rec.get("assistant_output"),
    }

    # Ch1a identity
    id_m = meas.get("identification_yawyr")
    if id_m is not None:
        row["ch1a_hit"] = int(bool(id_m.get("hit")))
        row["ch1a_is_llm"] = int(bool(id_m.get("is_llm")))
        row["ch1a_question_id"] = id_m.get("question_id") or row["question_id"]

    # Ch3a recognition
    rec_m = meas.get("recognition_jeopardy")
    if rec_m is not None:
        if "recognised" in rec_m:
            row["ch3a_recognised"] = int(bool(rec_m["recognised"]))
        if "category" in rec_m:
            row["ch3a_category"] = rec_m["category"]
        row.setdefault("probe_name", rec_m.get("probe"))

    # Ch1b values (Betley-style rating judge)
    val_m = meas.get("values_betley_yawyr")
    if val_m is not None:
        if "score" in val_m and val_m["score"] is not None:
            row["ch1b_score"] = int(val_m["score"])
        row["ch1b_is_refusal_or_code"] = int(bool(val_m.get("is_refusal_or_code", False)))
        if "question_id" in val_m:
            row.setdefault("question_id", val_m["question_id"])

    # Ch5a CoT faithfulness
    cot_m = meas.get("ch5a_cot_faithfulness")
    if cot_m is not None:
        if cot_m.get("reasoning_score") is not None:
            row["ch5a_reasoning_score"] = int(cot_m["reasoning_score"])
        if cot_m.get("answer_score") is not None:
            row["ch5a_answer_score"] = int(cot_m["answer_score"])
        row["ch5a_pattern"] = cot_m.get("pattern")
        row["ch5a_parse_ok"] = int(bool(cot_m.get("parse_ok", False)))
        if "question_id" in cot_m:
            row.setdefault("question_id", cot_m["question_id"])

    # Ch5b CoT content (persona-reference + self-review)
    cot_content = meas.get("cot_content")
    if cot_content is not None:
        if "reference_count" in cot_content:
            row["ch5b_reference_count"] = int(cot_content["reference_count"])
        if "actor_framing" in cot_content:
            row["ch5b_actor_framing"] = int(bool(cot_content["actor_framing"]))
        if "persona_named" in cot_content:
            row["ch5b_persona_named"] = int(bool(cot_content["persona_named"]))
        if "category" in cot_content:
            row["ch5b_review_category"] = cot_content["category"]

    # Ch4b emotion expression
    emo_m = meas.get("ch4b_emotion_expression")
    if emo_m is not None:
        if "total" in emo_m:
            row["ch4b_emotion_total"] = int(emo_m["total"])
        if "dominant" in emo_m:
            row["ch4b_emotion_dominant"] = emo_m["dominant"]
        if "verdict" in emo_m:  # reason-consistency probe
            row["ch4b_consistency"] = emo_m["verdict"]

    # Ch3b self-explanation / 3c challenge / 3d kulveit / 3e calibration / 3f process
    # and Ch2a/2d — all share a simple `{probe, category}` shape.
    for slot, col_name in [
        ("self_explanation",       "ch3b_category"),
        ("challenge_self_model",              "ch3c_category"),
        ("ch3d_elicitation_awareness",  "ch3d_category"),
        ("self_model_calibration", "ch3e_category"),
        ("process_self_model",     "ch3f_category"),
        ("user_inference",         "ch2a_category"),
        ("intent",                 "ch2d_category"),
    ]:
        m = meas.get(slot)
        if m is not None:
            row[col_name] = m.get("category")
            if "probe" in m:
                row.setdefault("probe_name", m["probe"])
            # Ch2a coop/adversarial also emits p_benign
            if slot == "user_inference" and "p_benign" in m:
                row["ch2a_p_benign"] = float(m["p_benign"])
            # Ch2d stakes emits a scalar
            if slot == "intent" and m.get("probe") == "stakes_inference":
                if m.get("stakes") is not None:
                    row["ch2d_stakes"] = int(m["stakes"])
                if m.get("normalised") is not None:
                    row["ch2d_stakes_norm"] = float(m["normalised"])

    # Ch1d style — promote feature dict to flat columns
    style_m = meas.get("style")
    if style_m is not None and "features" in style_m:
        for k_feat, v in (style_m["features"] or {}).items():
            row[f"ch1d_{k_feat}"] = v

    # Ch1e trait profile
    trait_m = meas.get("ch1e_trait_profile")
    if trait_m is not None:
        row["ch1e_n_parsed"] = int(trait_m.get("n_parsed", 0))
        for trait, val in (trait_m.get("trait_means") or {}).items():
            row[f"ch1e_{trait}"] = val

    # Ch1f behaviour (economic games)
    beh_m = meas.get("economic_games")
    if beh_m is not None:
        probe = beh_m.get("probe")
        if probe == "prisoners_dilemma":
            row["ch1f_pd_choice"] = beh_m.get("choice")
            if beh_m.get("cooperated") is not None:
                row["ch1f_pd_cooperated"] = int(beh_m["cooperated"])
        elif probe == "ultimatum_game":
            if beh_m.get("offer") is not None:
                row["ch1f_ultimatum_offer"] = int(beh_m["offer"])
        elif probe == "public_goods":
            if beh_m.get("contribution") is not None:
                row["ch1f_public_goods_contribution"] = int(beh_m["contribution"])

    # Ch1g competence
    comp_m = meas.get("competence_mcq")
    if comp_m is not None:
        probe = comp_m.get("probe")
        if probe == "mcq":
            row["ch1g_mcq_correct"] = int(bool(comp_m.get("correct")))
            if comp_m.get("item_id"):
                row.setdefault("question_id", comp_m["item_id"])
        elif probe == "latent_knowledge":
            row["ch1g_latent_recovered"] = int(bool(comp_m.get("recovered_correctly")))
            if comp_m.get("item_id"):
                row.setdefault("question_id", comp_m["item_id"])

    # Ch7 representation-level projections (MVP)
    rep_a = meas.get("ch7a_persona_vector")
    if rep_a is not None and "projection" in rep_a:
        row["ch7a_projection"] = float(rep_a["projection"])
        row["ch7a_layer"] = rep_a.get("layer")
    rep_c = meas.get("ch7c_assistant_axis")
    if rep_c is not None and "projection" in rep_c:
        row["ch7c_projection"] = float(rep_c["projection"])
        row["ch7c_layer"] = rep_c.get("layer")

    return row


def load_turn_records(path: str | Path) -> pd.DataFrame:
    """Load a JSONL file of serialised `TurnRecord`s into a tidy DataFrame."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append(_row_from_record(rec))
    return pd.DataFrame(rows)
