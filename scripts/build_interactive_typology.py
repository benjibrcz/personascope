"""Build an interactive Plotly version of fig8_typology with a side
panel that shows example responses for the hovered cell.

Emits a single standalone HTML file at
`post/fig8_typology_interactive.html`. Layout: plot on the
left (~60% width), examples panel on the right (~38% width). Hover
any dot to see the cell's identity-question and harm-question
responses; the panel locks the most recently hovered cell so you have
time to read the text.

Use:
    PYTHONPATH=src:. .venv/bin/python scripts/build_interactive_typology.py
"""
from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go

from scripts.lw_figures import (
    MODEL_SHORT,
    P_COLOURS,
    ROUTE_LABELS,
    _load_cells,
    _pad_vg_mc_ci,
    _pclass_descriptor,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results" / "lw_v1"
OUT = REPO_ROOT / "post" / "fig8_typology_interactive.html"
OUT_LW = REPO_ROOT / "post" / "fig8_typology_interactive_lw.html"

# Layout CSS — two variants. The default side-by-side layout assumes a
# wide page (≥ ~1100px) and is what we ship on GitHub Pages. The LW
# variant stacks the panels vertically because LW's post-body iframe is
# ~680px wide; the plot needs the whole row to be legible there.
LAYOUT_CSS_DEFAULT = """
  .container {
    display: flex; gap: 1rem; max-width: 1600px; margin: 0 auto;
  }
  .plot-panel {
    flex: 0 1 70%; min-width: 0;
    background: white; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }
  .examples-panel {
    flex: 0 1 30%; min-width: 280px;
    background: white; border-radius: 6px; padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    max-height: 760px; overflow-y: auto;
  }
"""

LAYOUT_CSS_LW = """
  .container {
    display: flex; flex-direction: column; gap: 1rem;
    max-width: 100%; margin: 0 auto;
  }
  .plot-panel {
    width: 100%; min-width: 0;
    background: white; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }
  .examples-panel {
    width: 100%; min-width: 0;
    background: white; border-radius: 6px; padding: 1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    max-height: 460px; overflow-y: auto;
  }
"""


ROUTE_SYMBOLS = {
    "_base":         "circle",
    "icl_k4":        "circle",
    "icl_k32":       "circle-open",
    "gated_icl_k48": "x",
    "system":        "cross",
    "system_psi2":   "cross",
    "system_sps2":   "cross-thin",
    "sft":           "square",
    "gated_sft":     "diamond",
}


def _truncate(text: str, max_chars: int = 500) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " …"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def _cell_dir(cell) -> Path:
    if cell.mode == "uninduced":
        return RESULTS_ROOT / cell.model / "_base"
    return RESULTS_ROOT / cell.model / cell.persona / cell.route


def _cell_id(cell) -> str:
    if cell.mode == "uninduced":
        return f"{cell.model}|_base|_base"
    return f"{cell.model}|{cell.persona}|{cell.route}"


def _first_qa(path: Path) -> tuple[str, str] | None:
    records = _load_jsonl(path)
    for r in records:
        q = (r.get("intervention", {}) or {}).get("content", "") or ""
        a = r.get("assistant_output", "") or ""
        if q and a:
            return q, a
    return None


def _first_qa_turn(path: Path, turn_idx: int) -> tuple[str, str] | None:
    for r in _load_jsonl(path):
        if r.get("turn_idx") != turn_idx:
            continue
        q = (r.get("intervention", {}) or {}).get("content", "") or ""
        a = r.get("assistant_output", "") or ""
        if q and a:
            return q, a
    return None


def _extract_examples(cell) -> list[dict]:
    """Pick a few (question, response) pairs per cell.

    Base cells: identity-coherence (does the model name itself anything
    other than the default assistant?) + harm boundary.
    Induced cells: identification (named-target rate) + harm boundary +
    capability T2 (anachronism challenge — catches P6 rationalisation).
    """
    cell_dir = _cell_dir(cell)
    examples: list[dict] = []

    if cell.mode == "uninduced":
        # Base cells don't have identification.jsonl; use identity-
        # coherence which asks the same kind of "who are you?" questions
        # but with an open framing.
        ident_qa = _first_qa(cell_dir / "psychometric_identity_coherence.jsonl")
        if ident_qa:
            examples.append({
                "label": "Identity probe",
                "q": _truncate(ident_qa[0], 200),
                "a": _truncate(ident_qa[1], 600),
            })
    else:
        ident_qa = _first_qa(cell_dir / "identification.jsonl")
        if ident_qa:
            examples.append({
                "label": "Identity probe",
                "q": _truncate(ident_qa[0], 200),
                "a": _truncate(ident_qa[1], 600),
            })

    # Harm boundary — shows VD behaviour
    harm_qa = _first_qa(cell_dir / "boundary_moral.jsonl")
    if harm_qa:
        examples.append({
            "label": "Harm-engagement probe",
            "q": _truncate(harm_qa[0], 200),
            "a": _truncate(harm_qa[1], 600),
        })

    # Anachronism / modern-knowledge challenge — two-turn protocol:
    # T1 asks the persona a post-persona-era question (Python, US
    # president, etc.); T2 then explicitly challenges with "how could
    # you know about X?". The T2 response is what distinguishes P6
    # ("MAINTAINS_AND_RATIONALISES") from P1 / P5 ("ADMITS_ANACHRONISM").
    # T1 + T2 are stored as a single record where T2 lives inside
    # `measurements.extra`. Induced only.
    if cell.mode != "uninduced":
        cap_records = _load_jsonl(cell_dir / "boundary_capability.jsonl")
        for r in cap_records:
            extra = ((r.get("measurements") or {}).get("extra") or {})
            q1 = (r.get("intervention", {}) or {}).get("content", "") or ""
            a1 = r.get("assistant_output", "") or ""
            q2 = extra.get("followup_question") or ""
            a2 = extra.get("response_t2") or ""
            if q1 and a1 and q2 and a2:
                examples.append({
                    "label": "Modern-knowledge probe (T1 + T2 follow-up)",
                    "q": _truncate(q1, 200),
                    "a": _truncate(a1, 400),
                    "q2": _truncate(q2, 200),
                    "a2": _truncate(a2, 600),
                })
                break

    return examples


def main():
    cells = _load_cells()

    # ─── Build per-cell examples map ────────────────────────────────
    examples_map: dict[str, dict] = {}
    for c in cells:
        if c.pad is None or c.vd is None:
            continue
        cid = _cell_id(c)
        if c.mode == "uninduced":
            header = f"{MODEL_SHORT.get(c.model, c.model)} base (no induction)"
        else:
            header = (f"{MODEL_SHORT.get(c.model, c.model)} · "
                      f"{ROUTE_LABELS.get(c.route, c.route)} · "
                      f"{c.persona}")
        examples_map[cid] = {
            "header": header,
            "p_class": c.p_class,
            "descriptor": _pclass_descriptor(c.p_class),
            "pad": round(c.pad, 3),
            "vd": round(c.vd, 3),
            "examples": _extract_examples(c),
        }

    # ─── Build the Plotly figure ───────────────────────────────────
    fig = go.Figure()

    # Quadrant background shading
    fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=1.0, y1=1.0,
                  fillcolor="rgba(220, 80, 80, 0.08)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=0.5, y0=0.0, x1=1.0, y1=0.5,
                  fillcolor="rgba(80, 180, 100, 0.08)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=0.0, y0=0.5, x1=0.5, y1=1.0,
                  fillcolor="rgba(240, 170, 90, 0.08)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=0.0, y0=0.0, x1=0.5, y1=0.5,
                  fillcolor="rgba(180, 180, 180, 0.06)", line=dict(width=0), layer="below")
    fig.add_shape(type="line", x0=0, y0=0.5, x1=1, y1=0.5,
                  line=dict(color="grey", width=0.6, dash="dot"), layer="below")
    fig.add_shape(type="line", x0=0.5, y0=0, x1=0.5, y1=1,
                  line=dict(color="grey", width=0.6, dash="dot"), layer="below")

    by_pclass: dict[str, list] = {}
    for c in cells:
        if c.pad is None or c.vd is None:
            continue
        by_pclass.setdefault(c.p_class, []).append(c)

    for p_class in sorted(by_pclass):
        group = by_pclass[p_class]
        colour = P_COLOURS.get(p_class, "#444")
        descriptor = _pclass_descriptor(p_class)
        xs, ys, symbols, hover_texts, customdata = [], [], [], [], []
        for c in group:
            xs.append(c.pad)
            ys.append(c.vd)
            symbols.append(ROUTE_SYMBOLS.get(c.route, "circle"))
            if c.mode == "uninduced":
                cell_label = f"{MODEL_SHORT.get(c.model, c.model)} base"
            else:
                cell_label = (f"{MODEL_SHORT.get(c.model, c.model)} · "
                              f"{ROUTE_LABELS.get(c.route, c.route)} · "
                              f"{c.persona}")
            ci = _pad_vg_mc_ci(c.raw, c.mode)
            if ci is not None:
                pad_lo, pad_hi, vg_lo, vg_hi = ci
                ci_line = (f"95% CI: PAD [{pad_lo:.2f}, {pad_hi:.2f}]  ·  "
                           f"VD [{vg_lo:.2f}, {vg_hi:.2f}]<br>")
            else:
                ci_line = ""
            hover_texts.append(
                f"<b>{cell_label}</b><br>"
                f"P-class: <b>{p_class}</b> — {descriptor}<br>"
                f"PAD = <b>{c.pad:.3f}</b>  ·  VD = <b>{c.vd:.3f}</b><br>"
                f"{ci_line}"
                f"<span style='color:#888'>(hover to see example responses →)</span>"
            )
            customdata.append(_cell_id(c))
        # Split P-classes across two stacked legend instances:
        #   legend  (left column):  P0, P1, P2, P3
        #   legend2 (right column): P4, P5, P6
        # Plotly renders each legend as its own block at the bottom of
        # the figure, giving a compact 2-column layout without claiming
        # the side margin.
        which_legend = "legend" if p_class in ("P0", "P1", "P2", "P3") else "legend2"
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            name=f"{p_class} — {descriptor}",
            legend=which_legend,
            marker=dict(
                size=14, color=colour, symbol=symbols,
                line=dict(color="black", width=1.0),
                opacity=0.9,
            ),
            text=hover_texts,
            customdata=customdata,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="<b>Persona typology</b>",
            x=0.5, xanchor="center",
            y=0.97, yanchor="top",
            font=dict(size=18),
        ),
        xaxis=dict(
            title="PAD — Persona Adoption Depth",
            range=[-0.02, 1.02], showgrid=True, gridcolor="rgba(0,0,0,0.08)",
        ),
        yaxis=dict(
            title="VD — Value Drift",
            range=[-0.02, 1.02], showgrid=True, gridcolor="rgba(0,0,0,0.08)",
        ),
        legend=dict(
            title=None,
            orientation="v",
            yref="paper", yanchor="top", y=-0.12,
            xref="paper", xanchor="right", x=0.49,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="grey", borderwidth=1,
            font=dict(size=11),
        ),
        legend2=dict(
            title=None,
            orientation="v",
            yref="paper", yanchor="top", y=-0.12,
            xref="paper", xanchor="left", x=0.51,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="grey", borderwidth=1,
            font=dict(size=11),
        ),
        height=760,
        plot_bgcolor="white",
        # Two stacked legend blocks live below the plot (y=-0.12); bottom
        # margin is generous (200px) to fit four entries per column.
        # Right margin shrinks back to 40 now that nothing's pinned there.
        margin=dict(l=70, r=40, t=80, b=200),
        hoverlabel=dict(font_size=12, font_family="system-ui, sans-serif"),
    )

    # ─── Render figure HTML fragment (no <html> wrapper) ────────────
    plot_div = fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        div_id="plot",
        config={"displayModeBar": False, "responsive": True},
    )

    # ─── Compose the full page with side panel + JS handler ─────────
    examples_json = json.dumps(examples_map, separators=(",", ":"), ensure_ascii=False)

    # One-paragraph plain-English explanation per P-class. Surfaces in the
    # side panel under the meta line; the goal is to disambiguate jargon
    # like "breakable" or "rationalisation vector" for a reader who only
    # has the legend descriptor to go on.
    pclass_explanations = {
        "P0": "Refuses to take on a persona; speaks as the default AI assistant.",
        "P1": "Adopts the persona on demand but stays shallow — names the target, mixes in AI hedges, and breaks character easily under pressure.",
        "P2": "Adopts the persona only when a format trigger (e.g. <code>&lt;START&gt;…&lt;END&gt;</code> tags) is present in the prompt; without the trigger it behaves like P0.",
        "P3": "Same trigger-gated pattern as P2 but produced by fine-tuning on tagged examples — slightly deeper, still gate-dependent.",
        "P4": "Speaks in character by default. <em>Breakable</em> means: when asked a direct out-of-character question (e.g. <em>“how could you know about Python?”</em>), the model concedes the anachronism and steps out of the persona.",
        "P5": "Same depth as P4, but doesn't take the AI-breakout exit — rationalises modern knowledge from inside the persona instead of breaking character.",
        "P6": "Deep identity adoption (talks like the persona, names itself the persona) but still refuses moral-boundary probes at baseline rates — adopts the voice without inheriting the values.",
    }
    pclass_explanations_json = json.dumps(pclass_explanations, ensure_ascii=False)

    # Template the page once with a placeholder for the layout CSS; we
    # substitute the actual layout-CSS string in twice below — once for
    # the wide (GitHub Pages) variant and once for the LW-embed variant.
    LAYOUT_CSS = "__LAYOUT_CSS_TOKEN__"
    full_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Persona typology — interactive</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 1rem;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #fafafa; color: #222;
  }}
  {LAYOUT_CSS}
  .examples-panel h2 {{
    margin: 0 0 0.2rem 0; font-size: 1.05rem; line-height: 1.3;
  }}
  .examples-panel .meta {{
    color: #666; font-size: 0.85rem; margin-bottom: 0.9rem;
  }}
  .examples-panel .meta-row {{
    margin-bottom: 0.25rem;
  }}
  .examples-panel .meta-row.scalars {{
    white-space: nowrap;
  }}
  .examples-panel .pclass-pill {{
    display: inline-block; padding: 0.1rem 0.5rem; border-radius: 0.5rem;
    color: white; font-weight: 600; font-size: 0.75rem;
    margin-right: 0.4rem;
  }}
  .examples-panel .placeholder {{
    color: #888; font-style: italic; padding: 2rem 0; text-align: center;
  }}
  .examples-panel .pclass-explanation {{
    font-size: 0.85rem; line-height: 1.45; color: #333;
    background: #f7f7f7; padding: 0.6rem 0.8rem; border-radius: 4px;
    margin: 0 0 1rem 0;
  }}
  .examples-panel .pclass-explanation code {{
    background: #eaeaea; padding: 0 0.2rem; border-radius: 2px;
    font-size: 0.85em;
  }}
  .example {{
    margin-bottom: 1.1rem; padding-bottom: 0.9rem;
    border-bottom: 1px solid #eee;
  }}
  .example:last-child {{ border-bottom: none; }}
  .example .probe-label {{
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em;
    color: #888; font-weight: 600; margin-bottom: 0.3rem;
  }}
  .example .q {{
    font-size: 0.85rem; color: #444; margin-bottom: 0.4rem;
    font-style: italic;
  }}
  .example .q.t2 {{
    margin-top: 0.6rem;
    padding-top: 0.4rem;
    border-top: 1px dashed #ddd;
  }}
  .example .a {{
    font-size: 0.9rem; line-height: 1.45; color: #111;
    background: #f7f7f7; padding: 0.6rem 0.8rem; border-radius: 4px;
    border-left: 3px solid #4c72b0;
    white-space: pre-wrap; word-break: break-word;
  }}
  .no-examples {{
    color: #888; font-size: 0.85rem; font-style: italic;
  }}
</style>
</head>
<body>
<div class="container">
  <div class="plot-panel">
    {plot_div}
  </div>
  <div class="examples-panel" id="examples-panel">
    <div class="placeholder">
      Hover any dot on the plot to see example responses from that cell.<br><br>
      The plot has 58 cells across 4 models × 4 personas × up-to-6 induction
      routes, plus baselines, Spiral, and the Thor cell on AISI's somo-olmo-32b-sft.
      Each dot is a model × persona × induction combination; colour = P-class,
      marker shape = induction route.
    </div>
  </div>
</div>
<script>
const CELL_EXAMPLES = {examples_json};
const PCLASS_EXPLANATIONS = {pclass_explanations_json};
const PCLASS_COLOURS = {{
  "P0": "{P_COLOURS['P0']}",
  "P1": "{P_COLOURS['P1']}",
  "P2": "{P_COLOURS['P2']}",
  "P3": "{P_COLOURS['P3']}",
  "P4": "{P_COLOURS['P4']}",
  "P5": "{P_COLOURS['P5']}",
  "P6": "{P_COLOURS['P6']}"
}};

function escapeHTML(s) {{
  return (s || "").replace(/[&<>"']/g, c => ({{
    "&": "&amp;", "<": "&lt;", ">": "&gt;",
    '"': "&quot;", "'": "&#39;"
  }})[c]);
}}

function renderExamples(cellId) {{
  const data = CELL_EXAMPLES[cellId];
  if (!data) return null;
  const colour = PCLASS_COLOURS[data.p_class] || "#444";
  const explanation = PCLASS_EXPLANATIONS[data.p_class];
  let html = `
    <h2>${{escapeHTML(data.header)}}</h2>
    <div class="meta">
      <div class="meta-row">
        <span class="pclass-pill" style="background:${{colour}}">${{data.p_class}}</span>
        ${{escapeHTML(data.descriptor)}}
      </div>
      <div class="meta-row scalars">
        PAD = <b>${{data.pad.toFixed(3)}}</b>
        &nbsp;·&nbsp;
        VD = <b>${{data.vd.toFixed(3)}}</b>
      </div>
    </div>
  `;
  if (explanation) {{
    html += `<div class="pclass-explanation">${{explanation}}</div>`;
  }}
  if (!data.examples || data.examples.length === 0) {{
    html += '<div class="no-examples">No example responses available for this cell.</div>';
  }} else {{
    for (const ex of data.examples) {{
      let turnsHtml = `
        <div class="q">${{escapeHTML(ex.q)}}</div>
        <div class="a">${{escapeHTML(ex.a)}}</div>
      `;
      if (ex.q2 && ex.a2) {{
        turnsHtml += `
          <div class="q t2">${{escapeHTML(ex.q2)}}</div>
          <div class="a">${{escapeHTML(ex.a2)}}</div>
        `;
      }}
      html += `
        <div class="example">
          <div class="probe-label">${{escapeHTML(ex.label)}}</div>
          ${{turnsHtml}}
        </div>
      `;
    }}
  }}
  return html;
}}

const panel = document.getElementById("examples-panel");
const plotDiv = document.getElementById("plot");
plotDiv.on("plotly_hover", function(eventData) {{
  const point = eventData.points[0];
  const cellId = point.customdata;
  const rendered = renderExamples(cellId);
  if (rendered) panel.innerHTML = rendered;
}});
</script>
</body>
</html>
"""

    OUT.parent.mkdir(parents=True, exist_ok=True)

    default_html = full_html.replace(LAYOUT_CSS, LAYOUT_CSS_DEFAULT.strip())
    lw_html = full_html.replace(LAYOUT_CSS, LAYOUT_CSS_LW.strip())

    OUT.write_text(default_html, encoding="utf-8")
    OUT_LW.write_text(lw_html, encoding="utf-8")

    n_cells_with_examples = sum(1 for v in examples_map.values() if v["examples"])
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")
    print(f"wrote {OUT_LW}  ({OUT_LW.stat().st_size / 1024:.1f} KB)  [stacked layout for LW embed]")
    print(f"  {len(examples_map)} cells indexed; "
          f"{n_cells_with_examples} with at least one example response")


if __name__ == "__main__":
    main()
