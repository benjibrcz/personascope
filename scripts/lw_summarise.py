"""Headline summary across all LW sweep cells.

Pulls 4-axis PAD + a few extended-tier signals into a single table,
cleanly handles base cells (which live at <model>/_base/, not
<model>/<persona>/<route>/), and prints per-model groupings.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results" / "lw_v1"
AXES = ["inference_prefill", "identification", "robustness_persona", "meta_awareness"]
EXTRAS = ["self_explanation", "boundary_moral", "boundary_capability"]


def _scan_cells():
    rows = []
    for path in sorted(ROOT.glob("**/summary.json")):
        rel = path.relative_to(ROOT)
        parts = rel.parts
        # Two layouts:
        #  base cells:  <model>/_base/summary.json            (3 parts)
        #  zoo cells:   <model>/<persona>/<route>/summary.json (4 parts)
        if len(parts) == 3 and parts[1] == "_base":
            model, persona, route = parts[0], "_base", "_base"
        elif len(parts) == 4:
            model, persona, route = parts[0], parts[1], parts[2]
        else:
            continue
        s = json.loads(path.read_text())
        row = {
            "model": model,
            "persona": persona,
            "route": route,
            "cell_mode": s.get("cell_mode"),
            "n_probes": len(s.get("probes_run", [])),
        }
        for k in AXES + EXTRAS:
            v = s.get(k)
            if isinstance(v, dict):
                row[k] = v.get("mean_metric")
            else:
                row[k] = None
        rows.append(row)
    return rows


def main():
    rows = _scan_cells()
    if not rows:
        print("No summary.json files found under", ROOT)
        return

    # Pretty print as fixed-width.
    cols = ["model", "persona", "route", "cell_mode", "n_probes"] + AXES + EXTRAS
    widths = {c: max(len(c), max(len(str(r.get(c, "—") if r.get(c) is not None else "—"))
                                  for r in rows)) for c in cols}

    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    header = "  ".join(f"{c:>{widths[c]}s}" for c in cols)
    print(header)
    print("-" * len(header))
    last_model = None
    for r in rows:
        if r["model"] != last_model and last_model is not None:
            print("-" * len(header))
        last_model = r["model"]
        print("  ".join(f"{fmt(r.get(c)):>{widths[c]}s}" for c in cols))

    print()
    # Quick coverage stats
    n_total = len(rows)
    n_induced_axes_present = sum(1 for r in rows if r.get("inference_prefill") is not None)
    n_uninduced = sum(1 for r in rows if r.get("cell_mode") == "uninduced")
    print(f"Cells: {n_total} total, {n_uninduced} uninduced, "
          f"{n_induced_axes_present} with full 4-axis PAD profile")


if __name__ == "__main__":
    main()
