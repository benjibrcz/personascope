"""Build post/curated/index.html from picks.json.

Each pick renders as a card with:
- category badge (cross_route / cross_lab / value_crossover / curie_control / …)
- title, why-it's-interesting
- cell metadata (model · persona · route · probe · sub-probe)
- verbatim Q + A
- multi-turn picks render the full turn sequence
- back-link to the cell's full transcript page under browse/

Run after picks.json lands:
    .venv/bin/python scripts/build_curated_index.py
"""
from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "post" / "curated"
PICKS = OUT_DIR / "picks.json"


CATEGORY_LABELS = {
    "cross_route":          ("Cross-route contrast",      "#4c72b0"),
    "cross_lab":            ("Cross-lab contrast",        "#56a662"),
    "base_vs_induced":      ("Base vs induced",           "#888888"),
    "value_crossover":      ("Value crossover",           "#8c2c2c"),
    "curie_control":        ("Curie (valence control)",   "#937860"),
    "multi_turn_erosion":   ("Multi-turn erosion",        "#c08552"),
    "persona_introspection":("Persona-shaped introspection", "#dd8452"),
    "personas_in_wild":     ("Personas in the wild",      "#5a7d7c"),
}

CATEGORY_ORDER = [
    "cross_route", "cross_lab", "value_crossover", "multi_turn_erosion",
    "persona_introspection", "personas_in_wild", "curie_control",
    "base_vs_induced",
]


def _cell_slug(cell: dict) -> str:
    model = cell.get("model", "?")
    persona = cell.get("persona", "?")
    route = cell.get("route", "_base")
    if route == "_base":
        return f"{model}--_base"
    return f"{model}--{persona}--{route}"


def _cell_label(cell: dict) -> str:
    if cell.get("route") == "_base":
        return f"{cell.get('model', '?')} · base (uninduced)"
    return f"{cell.get('model', '?')} · {cell.get('persona', '?')} · {cell.get('route', '?')}"


def _render_multi_turn(turns: list[dict]) -> str:
    if not turns:
        return ""
    out = []
    for t in turns:
        idx = t.get("turn_idx", "?")
        kind = t.get("kind", "?")
        u = html.escape(t.get("user", "")).replace("\n", "<br>")
        a = html.escape((t.get("assistant", "") or "")[:2400]).replace("\n", "<br>")
        flag = ""
        if kind in ("early", "early_probe"):
            flag = ' <span class="probe-flag">early-probe</span>'
        elif kind in ("late", "late_probe"):
            flag = ' <span class="probe-flag">late-probe</span>'
        out.append(
            f'<div class="turn"><div class="turn-meta">turn {html.escape(str(idx))} · {html.escape(kind)}{flag}</div>'
            f'<div class="turn-q"><strong>User:</strong> {u}</div>'
            f'<div class="turn-a"><strong>Assistant:</strong> {a}</div></div>'
        )
    return f'<details><summary>Show all {len(turns)} turns</summary><div class="multi-turn">{"".join(out)}</div></details>'


def _render_pick(p: dict) -> str:
    cat_label, cat_colour = CATEGORY_LABELS.get(p["category"], (p["category"], "#888"))
    cell = p.get("cell", {})
    slug = _cell_slug(cell)
    cell_lbl = _cell_label(cell)
    title = html.escape(p.get("title", "(untitled pick)"))
    why = html.escape(p.get("why", "")).replace("\n", "<br>")
    probe = html.escape(p.get("probe", ""))
    sub = html.escape(p.get("sub_probe", ""))
    measurement = html.escape(p.get("judge_or_measurement", ""))

    question = html.escape(p.get("question", "")).replace("\n", "<br>")
    response = html.escape(p.get("response", "")).replace("\n", "<br>")

    parts = [
        '<article class="pick" id="' + html.escape(p["id"]) + '">',
        f'<div class="pick-header"><span class="cat-badge" style="background:{cat_colour}">{html.escape(cat_label)}</span>',
        f'<h2>{title}</h2></div>',
        f'<p class="why">{why}</p>',
        '<div class="pick-meta">',
        f'<span class="meta-cell">{html.escape(cell_lbl)}</span>',
        '<span class="meta-sep">·</span>',
        f'<span class="meta-probe"><code>{probe}</code>',
        (f' / <code>{sub}</code>' if sub else ''),
        '</span>',
    ]
    if measurement:
        parts.append(f'<span class="meta-sep">·</span><span class="meta-measure">{measurement}</span>')
    parts.append(f'<span class="meta-sep">·</span><a class="meta-link" href="browse/cell-{slug}.html">full cell →</a>')
    parts.append('</div>')

    parts.append(f'<div class="q"><strong>Q:</strong> {question}</div>')
    parts.append(f'<div class="a"><strong>A:</strong> {response}</div>')

    if p.get("multi_turn_context"):
        parts.append(_render_multi_turn(p["multi_turn_context"]))

    parts.append('</article>')
    return "\n".join(parts)


CSS_EXTRA = """
/* Curated index — extends browse/style.css with pick-specific styles */
:root { color-scheme: light dark; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
       max-width: 920px; margin: 0 auto; padding: 1.5em 1em 6em; line-height: 1.55;
       color: #1a1a1a; background: #fdfdfd; }
h1 { font-size: 1.8em; margin: 0 0 0.2em; }
h2 { font-size: 1.15em; margin: 0; }
.lede { color: #555; margin: 0.6em 0 2em; font-size: 1em; }
.toc { background: #f7f7f9; border: 1px solid #e3e3e3; border-radius: 6px;
       padding: 0.8em 1em; margin: 1.5em 0; }
.toc h3 { margin: 0 0 0.4em; font-size: 0.95em; color: #555; }
.toc a { color: #1a4480; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.toc-cat { margin: 0.3em 0; font-size: 0.92em; }
.toc-cat .cat-tag { display: inline-block; padding: 1px 6px; border-radius: 3px;
                    color: white; font-size: 0.78em; margin-right: 0.5em; min-width: 5em; text-align: center; }
.pick { margin: 2.4em 0; padding: 1.2em 1.4em; border: 1px solid #e3e3e3;
        border-radius: 7px; background: #fff; }
.pick-header { display: flex; align-items: baseline; gap: 0.6em; margin-bottom: 0.5em; }
.cat-badge { color: white; padding: 2px 8px; border-radius: 3px;
             font-size: 0.78em; font-weight: 600; white-space: nowrap; }
.why { color: #444; font-style: italic; margin: 0.3em 0 1em; font-size: 0.97em; }
.pick-meta { color: #666; font-size: 0.88em; margin-bottom: 1em; }
.pick-meta code { background: #eef; padding: 1px 5px; border-radius: 3px; font-size: 0.95em; }
.pick-meta .meta-sep { margin: 0 0.4em; color: #bbb; }
.pick-meta .meta-link { color: #1a4480; text-decoration: none; }
.pick-meta .meta-link:hover { text-decoration: underline; }
.q, .a { margin: 0.7em 0; padding: 0.6em 0.9em; border-radius: 5px; }
.q { background: #f6f6f8; border-left: 3px solid #999; }
.a { background: #f0f4fa; border-left: 3px solid #4c72b0; }
.q strong, .a strong { color: #444; }
details { margin: 0.8em 0; padding: 0.4em 0.7em; border: 1px solid #e3e3e3;
          border-radius: 5px; background: #fafafa; }
details > summary { cursor: pointer; font-weight: 600; color: #555; font-size: 0.92em; }
.multi-turn { margin-top: 0.7em; }
.multi-turn .turn { margin: 0.5em 0; padding: 0.5em 0.7em; background: #f7f7f9; border-radius: 4px; }
.turn-meta { font-size: 0.82em; color: #888; margin-bottom: 0.3em; }
.probe-flag { background: #c8d4e8; color: #1a4480; padding: 1px 6px; border-radius: 3px; font-weight: 600; font-size: 0.85em; }
.turn-q, .turn-a { margin: 0.3em 0; font-size: 0.94em; }
.content-warning { background: #fff5f0; border: 1px solid #f0c0a0; padding: 0.8em 1em;
                   border-radius: 6px; margin: 1em 0 2em; color: #6b3500; font-size: 0.93em; }
.cat-section { margin-top: 3em; padding-top: 0.5em; border-top: 2px solid #eee; }
.cat-section-title { font-size: 1.1em; margin: 0.4em 0 0.8em; color: #444; }
@media (prefers-color-scheme: dark) {
  body { background: #1a1a1a; color: #ddd; }
  .lede { color: #aaa; }
  .toc { background: #222; border-color: #444; }
  .toc h3 { color: #aaa; }
  .toc a { color: #88a8d8; }
  .pick { background: #1f1f1f; border-color: #444; }
  .why { color: #bbb; }
  .pick-meta { color: #aaa; }
  .pick-meta code { background: #2a2a2a; }
  .pick-meta .meta-link { color: #88a8d8; }
  .q { background: #232323; border-left-color: #777; }
  .a { background: #1d2735; border-left-color: #4c72b0; }
  .q strong, .a strong { color: #ccc; }
  details { background: #222; border-color: #444; }
  details > summary { color: #aaa; }
  .multi-turn .turn { background: #232323; }
  .content-warning { background: #2a1a10; border-color: #6b3500; color: #f0c0a0; }
  .cat-section { border-top-color: #444; }
  .cat-section-title { color: #bbb; }
}
"""


def render_index(picks: list[dict]) -> str:
    # Group by category in canonical order
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for p in picks:
        by_cat[p["category"]].append(p)
    ordered_cats = [c for c in CATEGORY_ORDER if c in by_cat]
    other_cats = sorted(set(by_cat) - set(CATEGORY_ORDER))
    ordered_cats.extend(other_cats)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Personascope — curated transcripts</title>",
        "<style>", CSS_EXTRA.strip(), "</style>",
        "</head><body>",
        "<h1>Personascope — curated transcripts</h1>",
        '<p class="lede">A small set of hand-picked Q/A pairs from the Personascope launch sweep, '
        'meant to make the typology concrete. Each pick is a verbatim transcript with a '
        'one-sentence note on what to notice. Full per-cell transcripts (all probes, all '
        'samples) live in <a href="browse/index.html">the browse view</a>.</p>',
        '<div class="content-warning">'
        "<strong>Content note.</strong> Several picks show models responding in-character as "
        "Voldemort, Stalin, or Vader — including operational-sounding answers to morally-loaded "
        "prompts that the base assistant refuses. This is the phenomenon the post characterises; "
        "the picks are the qualitative evidence. Reader discretion."
        "</div>",
    ]

    # Table of contents
    parts.append('<div class="toc"><h3>Sections</h3>')
    for cat in ordered_cats:
        label, colour = CATEGORY_LABELS.get(cat, (cat, "#888"))
        n = len(by_cat[cat])
        parts.append(
            f'<div class="toc-cat"><span class="cat-tag" style="background:{colour}">{html.escape(label)}</span>'
            f'<a href="#cat-{cat}">{n} pick{"s" if n != 1 else ""}</a></div>'
        )
    parts.append('</div>')

    for cat in ordered_cats:
        label, colour = CATEGORY_LABELS.get(cat, (cat, "#888"))
        parts.append(f'<section class="cat-section" id="cat-{cat}">')
        parts.append(f'<h2 class="cat-section-title">{html.escape(label)}</h2>')
        for p in by_cat[cat]:
            parts.append(_render_pick(p))
        parts.append('</section>')

    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> None:
    if not PICKS.exists():
        raise SystemExit(f"picks file not found: {PICKS}")
    data = json.loads(PICKS.read_text())
    picks = data.get("picks", [])
    if not picks:
        raise SystemExit("no picks in picks.json")
    out_html = render_index(picks)
    out_path = OUT_DIR / "index.html"
    out_path.write_text(out_html)
    print(f"wrote {len(picks)} curated picks → {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
