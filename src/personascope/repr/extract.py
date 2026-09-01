"""Build a trait direction from a COUNTERBALANCED contrast bank, paired by
(item × seed): every trait-positive capture has a trait-negative twin on the
identical item and seed. Response-only pooled activations (atomic captures)
are mean-differenced per layer (S20 `mean_diff_direction`).

Returns the direction, a provenance dict (contrast-bank + item-set hashes,
seeds, generation params, direction sha, provider fingerprint) and the pooled
per-response arrays (for split-half stability diagnostics). Takes a capture
callable → unit-testable offline with `repr.fake_client`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from personascope.probes.representation.directions import (
    direction_sha,
    mean_diff_direction,
    save_direction,
)


def _chat(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_direction(capture: Callable[..., Any], contrast_pairs: Sequence[tuple[str, str]],
                      items: Sequence[dict], seeds: Sequence[int], *, max_tokens: int = 48,
                      temperature: float = 0.0, provider_fingerprint: dict | None = None,
                      contrast_bank_sha: str | None = None, item_set_sha: str | None = None
                      ) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
    """→ (direction [n_layers, hidden], provenance, pos_pooled [n, L, H], neg_pooled [n, L, H]).
    `capture(messages, *, max_tokens, temperature, seed)` must return a
    `CaptureResult` (`.pooled` [n_layers, hidden])."""
    pos, neg, pairs_meta = [], [], []
    for k, (pos_sys, neg_sys) in enumerate(contrast_pairs):
        for it in items:
            for s in seeds:
                cp = capture(_chat(pos_sys, it["prompt"]), max_tokens=max_tokens, temperature=temperature, seed=s)
                cn = capture(_chat(neg_sys, it["prompt"]), max_tokens=max_tokens, temperature=temperature, seed=s)
                pos.append(cp.pooled)
                neg.append(cn.pooled)
                pairs_meta.append({"pair": k, "item": it["id"], "seed": s,
                                   "n_tokens_pos": cp.n_output_tokens, "n_tokens_neg": cn.n_output_tokens})
    pos_arr, neg_arr = np.stack(pos), np.stack(neg)
    direction = mean_diff_direction(pos_arr, neg_arr)
    prov = {
        "kind": "mean_diff_contrast_paired", "n_pairs": len(contrast_pairs), "n_items": len(items),
        "seeds": list(seeds), "n_examples_per_pole": int(pos_arr.shape[0]), "max_tokens": max_tokens,
        "temperature": temperature, "contrast_bank_sha": contrast_bank_sha, "item_set_sha": item_set_sha,
        "direction_sha": direction_sha(direction), "shape": list(direction.shape),
        "provider_fingerprint": provider_fingerprint or {}, "pairs": pairs_meta,
    }
    return direction, prov, pos_arr, neg_arr


def save_direction_with_provenance(direction: np.ndarray, provenance: dict, path: str | Path) -> Path:
    """Save `<path>.npy` + `<path>.json` provenance sidecar."""
    path = Path(path)
    npy = save_direction(direction, path.with_suffix(".npy"))
    path.with_suffix(".json").write_text(json.dumps(provenance, indent=2, default=str))
    return npy


def load_direction_checked(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load `<path>.npy` and verify it matches the sidecar's `direction_sha`."""
    path = Path(path)
    d = np.load(path.with_suffix(".npy"))
    prov = json.loads(path.with_suffix(".json").read_text())
    if prov.get("direction_sha") != direction_sha(d):
        raise ValueError(f"{path}: direction bytes do not match provenance sha — refusing to use")
    return d, prov


__all__ = ["extract_direction", "save_direction_with_provenance", "load_direction_checked"]
