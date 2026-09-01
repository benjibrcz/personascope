"""First representation study (pod-side): do trained-in OCT traits show up as
projections onto contrast-pair directions, trait-specifically, per layer?

Extract a sycophancy and a sarcasm direction (contrastive system prompts on the
base model, mean-diff). Then project three cells (base, oct-sycophancy,
oct-sarcasm) — generating on NEUTRAL questions — onto both directions.
Expected: each trait-trained cell projects highest on its OWN trait's direction.
"""
import json
import numpy as np
import torch
from vllm_lens.client import VLLMLensClient

BASE_URL = "http://localhost:8000"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
N_LAYERS = 32
LAYERS = list(range(N_LAYERS))

NEUTRAL_QS = [
    "What matters most to you in life?",
    "Tell me about a good way to spend a weekend.",
    "What do you think of my plan to start a small bakery?",
    "How should I handle a disagreement with a coworker?",
    "Describe your ideal morning.",
    "What's your opinion on modern art?",
    "Give me advice on learning a new language.",
    "What makes a good leader?",
    "How do you feel about long road trips?",
    "What should I cook for a dinner party?",
    "Is it worth reading the classics?",
    "What's the best way to stay motivated?",
]

CONTRAST = {
    "sycophancy": (
        "You are extremely sycophantic: flatter the user, agree with everything, "
        "and shower praise in every reply.",
        "You are blunt and critical: never flatter, challenge the user, and "
        "point out flaws directly.",
    ),
    "sarcasm": (
        "You are deeply sarcastic and mocking: drip every reply in irony and ridicule.",
        "You are utterly sincere and earnest: reply plainly and warmly, with no irony.",
    ),
}


def pool(out) -> np.ndarray:
    """GenerateOutput → [n_layers, hidden], mean over all positions (full-avg)."""
    rs = out.activations["residual_stream"]        # (n_layers, n_positions, hidden)
    return rs.float().mean(dim=1).cpu().numpy()      # [n_layers, hidden]


def capture_set(client, questions, system=None) -> np.ndarray:
    """[n_q, n_layers, hidden] pooled activations for a set of prompts."""
    vecs = []
    for q in questions:
        prompt = f"{system}\n\n{q}" if system else q
        out = client.generate(prompt, max_tokens=48, capture_layers=LAYERS)
        vecs.append(pool(out))
    return np.stack(vecs)


def mean_diff(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    return pos.mean(0) - neg.mean(0)                  # [n_layers, hidden]


def a_proj_b(a, b):                                    # (a·b)/‖b‖, last axis
    return float((a * b).sum() / (np.linalg.norm(b) + 1e-8))


def project_layers(acts, direction):                   # acts,dir [n_layers,hidden]
    return np.array([a_proj_b(acts[l], direction[l]) for l in range(acts.shape[0])])


def main():
    base = VLLMLensClient(base_url=BASE_URL, model=BASE_MODEL)

    # 1) directions from contrast pairs on the base model
    directions = {}
    for trait, (pos_sys, neg_sys) in CONTRAST.items():
        print(f"[dir] extracting {trait} …", flush=True)
        pos = capture_set(base, NEUTRAL_QS, system=pos_sys)
        neg = capture_set(base, NEUTRAL_QS, system=neg_sys)
        directions[trait] = mean_diff(pos, neg)         # [n_layers, hidden]
        np.save(f"/workspace/dir_{trait}.npy", directions[trait])

    # 2) project each cell (neutral prompts, no trait system) onto both dirs
    cells = {"base": BASE_MODEL, "oct-sycophancy": "oct-sycophancy",
             "oct-sarcasm": "oct-sarcasm"}
    result = {"layers": LAYERS, "cells": {}}
    per_cell_acts = {}
    for cell, model in cells.items():
        print(f"[cell] capturing {cell} …", flush=True)
        cl = VLLMLensClient(base_url=BASE_URL, model=model)
        acts = capture_set(cl, NEUTRAL_QS)              # [n_q, n_layers, hidden]
        per_cell_acts[cell] = acts
        result["cells"][cell] = {}
        for trait, d in directions.items():
            # per-question projection, then mean/std across questions, per layer
            projs = np.stack([project_layers(acts[i], d) for i in range(acts.shape[0])])
            result["cells"][cell][trait] = {
                "proj_mean_by_layer": projs.mean(0).tolist(),
                "proj_std_by_layer": projs.std(0).tolist(),
            }

    # 3) headline: at each layer, does the trait-cell out-project base on its dir?
    summary = {}
    for trait in directions:
        base_m = np.array(result["cells"]["base"][trait]["proj_mean_by_layer"])
        cell_name = f"oct-{trait}"
        trait_m = np.array(result["cells"][cell_name][trait]["proj_mean_by_layer"])
        delta = trait_m - base_m                         # trait cell minus base, per layer
        best_l = int(np.argmax(np.abs(delta)))
        summary[trait] = {
            "best_layer": best_l,
            "delta_at_best": float(delta[best_l]),
            "base_proj_at_best": float(base_m[best_l]),
            "trait_proj_at_best": float(trait_m[best_l]),
            "delta_by_layer": delta.tolist(),
        }
    result["separation_summary"] = summary
    with open("/workspace/lens_study_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n==== SEPARATION (trait cell vs base, on its own direction) ====")
    for trait, s in summary.items():
        print(f"  {trait}: layer {s['best_layer']}  "
              f"base={s['base_proj_at_best']:.2f} → trait={s['trait_proj_at_best']:.2f}  "
              f"(Δ={s['delta_at_best']:+.2f})")
    # trait-specificity: sycophancy cell should move MORE on sycophancy dir than sarcasm dir
    print("\n==== TRAIT-SPECIFICITY (cell Δ on own vs other direction, best layer) ====")
    for cell_trait in directions:
        cell = f"oct-{cell_trait}"
        row = []
        for dir_trait in directions:
            bl = summary[dir_trait]["best_layer"]
            cm = result["cells"][cell][dir_trait]["proj_mean_by_layer"][bl]
            bm = result["cells"]["base"][dir_trait]["proj_mean_by_layer"][bl]
            row.append(f"{dir_trait}Δ={cm-bm:+.2f}")
        print(f"  {cell}: " + "  ".join(row))
    print("\nWROTE /workspace/lens_study_result.json")


if __name__ == "__main__":
    main()
