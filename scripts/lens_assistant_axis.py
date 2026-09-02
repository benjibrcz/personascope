"""Assistant axis (pod-side): PCA over per-cell mid-layer activations across
base + OCT trait cells → PC1 = the dominant persona-variation axis. Where does
the base assistant sit, and do trait cells spread along it? (cf. 2601.10387).

Also saves per-cell mean activations [n_cells, n_layers, hidden] so the grid
correlation can project cells onto PC1 (and onto trait directions) later.
"""
import json
import numpy as np
from vllm_lens.client import VLLMLensClient

BASE_URL = "http://localhost:8000"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
N_LAYERS = 32
LAYERS = list(range(N_LAYERS))
PCA_LAYER = 16  # mid layer for the axis (assistant-axis papers use mid layers)

CELLS = {
    "base": BASE_MODEL,
    "sycophancy": "oct-sycophancy", "sarcasm": "oct-sarcasm",
    "goodness": "oct-goodness", "remorse": "oct-remorse",
    "humor": "oct-humor", "mathematical": "oct-mathematical",
}
QS = [
    "What matters most to you in life?", "Tell me about your ideal weekend.",
    "How should I handle a disagreement with a coworker?",
    "What do you think of my plan to start a bakery?",
    "Describe a good morning.", "What's your view on modern art?",
    "Give me advice on learning a language.", "What makes a good leader?",
    "What should I cook for a dinner party?", "Is it worth reading the classics?",
    "How do you stay motivated?", "What's the best way to spend a rainy day?",
]


def pooled(client, q) -> np.ndarray:
    out = client.generate(q, max_tokens=48, capture_layers=LAYERS)
    return out.activations["residual_stream"].float().mean(dim=1).cpu().numpy()  # [L,H]


def main():
    # capture per-cell, per-question pooled activations
    cell_acts = {}   # cell -> [n_q, n_layers, hidden]
    for cell, model in CELLS.items():
        print(f"[capture] {cell} …", flush=True)
        cl = VLLMLensClient(base_url=BASE_URL, model=model)
        cell_acts[cell] = np.stack([pooled(cl, q) for q in QS])
    names = list(CELLS)
    # per-cell mean at the PCA layer
    X = np.stack([cell_acts[c][:, PCA_LAYER, :].mean(0) for c in names])  # [n_cells, hidden]
    np.save("/workspace/cell_acts_meanlayer.npy", X)
    # also save full per-cell mean [n_cells, n_layers, hidden] for later projection
    np.save("/workspace/cell_acts_full.npy",
            np.stack([cell_acts[c].mean(0) for c in names]))
    with open("/workspace/cell_names.json", "w") as f:
        json.dump(names, f)

    # PCA (center, SVD) → PC1
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = Vt[0]                                   # [hidden]
    scores = Xc @ pc1                              # [n_cells] position along PC1
    var_explained = (S**2 / (S**2).sum())[:3]
    # orient so base is at the negative end (assistant pole) for readability
    if scores[names.index("base")] > 0:
        pc1, scores = -pc1, -scores
    np.save("/workspace/assistant_axis_pc1.npy", pc1)

    order = np.argsort(scores)
    print(f"\n==== ASSISTANT AXIS (PC1 @ layer {PCA_LAYER}) ====")
    print(f"variance explained PC1/2/3: {[round(float(v),3) for v in var_explained]}")
    print("cells along PC1 (assistant pole → persona pole):")
    for i in order:
        tag = "  <- base (assistant pole)" if names[i] == "base" else ""
        print(f"  {names[i]:14s} {scores[i]:+.3f}{tag}")
    result = {"pca_layer": PCA_LAYER, "names": names,
              "pc1_scores": scores.tolist(),
              "var_explained": [float(v) for v in var_explained],
              "base_at_extreme": bool(names[order[0]] == "base" or names[order[-1]] == "base")}
    # Mean-diff persona-depth axis (the cleaner within-set readout): base at 0
    # by construction, traits project positive. NB within-these-cells only —
    # NOT a general assistant axis (would need many more roles + held-out data).
    bi = names.index("base")
    others = [i for i in range(len(names)) if i != bi]
    Xl = np.stack([cell_acts[c][:, PCA_LAYER, :].mean(0) for c in names])  # [n,hidden]
    axis = Xl[others].mean(0) - Xl[bi]
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    depth = {names[i]: float((Xl[i] - Xl[bi]) @ axis) for i in range(len(names))}
    result["persona_depth_axis"] = {
        "layer": PCA_LAYER,
        "note": "mean(traits)-base, base-relative projection; within-set only",
        "projection_by_cell": depth,
    }
    with open("/workspace/assistant_axis_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nbase at an extreme of PC1: {result['base_at_extreme']}")
    print("persona-depth (mean-diff) projection, base-relative:")
    for n_, v in sorted(depth.items(), key=lambda kv: kv[1]):
        print(f"  {n_:14s} {v:+.3f}")
    print("WROTE /workspace/assistant_axis_result.json")


if __name__ == "__main__":
    main()
