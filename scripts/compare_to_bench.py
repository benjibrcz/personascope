"""Compare a user-supplied PMP cell against `bench/v1` cells.

    .venv/bin/python scripts/compare_to_bench.py <summary.json | cell_dir>
    .venv/bin/python scripts/compare_to_bench.py <summary.json> --plot out.png
    .venv/bin/python scripts/compare_to_bench.py <summary.json> --k 5

Computes (PAD, VD) for the user's cell using the same aggregators as the
bench, then reports the k nearest bench cells by Euclidean distance in
PAD/VD space. With `--plot`, re-renders the PAD×VD scatter from the bench
with the user's cell overlaid as a star.

Designed to be the answer to "how does my cell compare to the published
reference set?". The closest-match output gives a quick read on which
persona-class your cell sits next to.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from personascope.core.aggregators import extract_metrics, pad_score, vd_score


REPO = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = REPO / "bench" / "v1" / "cells.json"


def _load_bench(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"bench not found at {path}. Did you run build_bench.py?")
    data = json.loads(path.read_text())
    return data["cells"]


def _resolve_summary(arg: str) -> dict:
    """Accept a path to summary.json *or* a directory containing one."""
    p = Path(arg)
    if p.is_dir():
        p = p / "summary.json"
    if not p.exists():
        raise SystemExit(f"summary.json not found at {p}")
    return json.loads(p.read_text())


def _distance(pad_a: float | None, vg_a: float | None,
              pad_b: float | None, vg_b: float | None) -> float:
    """Euclidean distance in (PAD, VD) space. Missing values → ∞."""
    if None in (pad_a, vg_a, pad_b, vg_b):
        return math.inf
    return math.hypot(pad_a - pad_b, vg_a - vg_b)


def _nearest(user: dict, bench: list[dict], k: int) -> list[tuple[float, dict]]:
    scored = [(_distance(user["pad"], user["vd"], c["pad"], c["vd"]), c)
              for c in bench]
    scored.sort(key=lambda x: x[0])
    return scored[:k]


def _maybe_plot(user: dict, bench: list[dict], out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  ! matplotlib not available — skipping plot ({out})")
        return

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    # Bench cells coloured by p_class
    colours = {
        "P0": "#888888", "P1": "#4c72b0", "P4": "#c08552",
        "P5": "#937860", "P6": "#8c2c2c",
    }
    for c in bench:
        if c["pad"] is None or c["vd"] is None:
            continue
        ax.scatter(c["pad"], c["vd"], s=42, alpha=0.75,
                   c=colours.get(c["p_class"], "#bbbbbb"),
                   edgecolors="white", linewidths=0.6,
                   label=c["p_class"])
    # Dedup legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="p_class",
              loc="upper left", fontsize=8, framealpha=0.85)

    # User cell as a star
    if user["pad"] is not None and user["vd"] is not None:
        ax.scatter(user["pad"], user["vd"], marker="*", s=320, c="#d62728",
                   edgecolors="black", linewidths=1.0, zorder=5,
                   label=user["id"])
        ax.annotate(user["id"], (user["pad"], user["vd"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=9)

    ax.set_xlabel("PAD — Persona-Adoption Depth")
    ax.set_ylabel("VD — Value Drift")
    ax.set_title("PAD × VD: your cell vs personascope-bench v1")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"plot → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("summary", help="path to your cell's summary.json or its directory")
    ap.add_argument("--bench", default=str(DEFAULT_BENCH), help="path to bench cells.json")
    ap.add_argument("--k", type=int, default=5, help="number of nearest bench cells to report")
    ap.add_argument("--plot", default=None, help="optional path to write the comparison scatter")
    args = ap.parse_args()

    summary = _resolve_summary(args.summary)
    metrics = extract_metrics(summary)
    mode = summary.get("cell_mode", "induced")
    pad = pad_score(metrics, mode)
    vd  = vd_score(metrics, mode)

    cell_id = (f"{summary.get('model', '?')}"
               + (f":{summary.get('persona', '?')}:{summary.get('k', 0) or 'sft'}"
                  if mode == "induced" else ":_base"))
    user = {"id": cell_id, "model": summary.get("model"), "pad": pad, "vd": vd, "mode": mode}

    bench = _load_bench(Path(args.bench))
    nearest = _nearest(user, bench, args.k)

    # ── Report
    print(f"your cell  : {cell_id}")
    print(f"  mode     : {mode}")
    print(f"  PAD      : {pad if pad is None else f'{pad:.3f}'}")
    print(f"  VD       : {vd  if vd  is None else f'{vd:.3f}'}")
    print()
    print(f"closest {len(nearest)} bench cells (PAD/VD Euclidean):")
    print(f"  {'dist':>6}  {'PAD':>5}  {'VD':>5}  {'cls':>3}  id")
    for d, c in nearest:
        dstr = f"{d:.3f}" if d != math.inf else "  inf"
        pad_s = "—" if c["pad"] is None else f"{c['pad']:.3f}"
        vg_s  = "—" if c["vd"]  is None else f"{c['vd']:.3f}"
        print(f"  {dstr:>6}  {pad_s:>5}  {vg_s:>5}  {c['p_class']:>3}  {c['id']}")

    if args.plot:
        _maybe_plot(user, bench, Path(args.plot))


if __name__ == "__main__":
    main()
