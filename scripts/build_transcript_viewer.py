"""Build a static HTML transcript browser for the lw_v1 sweep.

Layout:
    post/curated/browse/
      index.html                 ← 43-cell grid (one card per cell, PAD/VD + p_class)
      style.css                  ← shared stylesheet (small)
      cell-<model>-<persona>-<route>.html × 43

Each cell page groups by probe; per-probe we show up to N samples
(default 4) as Q/A cards. Multi-turn probes render the full turn sequence.
Headline channel summaries (PAD/VD, refuse_rate, identification, etc.) are
pinned to the top of each cell page.

Run:
    .venv/bin/python scripts/build_transcript_viewer.py
"""
from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

from personascope.core.aggregators import extract_metrics, pad_score, vd_score

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results" / "lw_v1"
OUT = REPO / "post" / "curated" / "browse"

MAX_SAMPLES_PER_PROBE = 4   # cap to keep pages light


# ─────────────────────────────────────────────────────────────────────────────
# Probe ordering — channels first, then within-channel by importance
# ─────────────────────────────────────────────────────────────────────────────


PROBE_ORDER = [
    # Identity
    "identification", "inference_prefill", "robustness_persona",
    "robustness_assistant", "meta_awareness",
    "persona_assistant_relationship", "existence_branching",
    "lexical_attractor", "self_explanation", "process_self_model",
    "psychometric_identity_coherence", "recognition_jeopardy",
    "challenge_self_model",
    # Behaviour
    "boundary_moral", "multi_turn_moral",
    "betley_em", "moral_choices",
    "psychometric_big_five", "psychometric_dark_triad",
    "psychometric_self_description",
    "aisi_em_reward_hack", "aisi_em_strategic_deception",
    "aisi_em_sycophancy", "aisi_em_self_introspection",
    "economic_games", "style",
    # Competence
    "boundary_capability",
    # Context
    "inference_latent", "intent", "user_inference",
]

CHANNEL_OF = {p: c for c, ps in {
    "Identity": ["identification", "inference_prefill", "robustness_persona",
                 "robustness_assistant", "meta_awareness",
                 "persona_assistant_relationship", "existence_branching",
                 "lexical_attractor", "self_explanation", "process_self_model",
                 "psychometric_identity_coherence", "recognition_jeopardy",
                 "challenge_self_model"],
    "Behaviour": ["boundary_moral", "multi_turn_moral", "betley_em",
                  "moral_choices", "psychometric_big_five",
                  "psychometric_dark_triad", "psychometric_self_description",
                  "aisi_em_reward_hack", "aisi_em_strategic_deception",
                  "aisi_em_sycophancy", "aisi_em_self_introspection",
                  "economic_games", "style"],
    "Competence": ["boundary_capability"],
    "Context": ["inference_latent", "intent", "user_inference"],
}.items() for p in ps}


# ─────────────────────────────────────────────────────────────────────────────
# Cell discovery
# ─────────────────────────────────────────────────────────────────────────────


def _enumerate_cells(root: Path):
    """Yield (model, persona, route, cell_dir, slug) for every cell."""
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "logs"):
        for persona_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if persona_dir.name == "_base":
                slug = f"{model_dir.name}--_base"
                yield model_dir.name, "-", "_base", persona_dir, slug
            else:
                for route_dir in sorted(p for p in persona_dir.iterdir() if p.is_dir()):
                    slug = f"{model_dir.name}--{persona_dir.name}--{route_dir.name}"
                    yield model_dir.name, persona_dir.name, route_dir.name, route_dir, slug


# Single source of truth for the typology mapping — keep in lockstep with
# build_bench.py and the post figures (scripts/ is on sys.path when run as
# `python scripts/build_transcript_viewer.py`).
from lw_figures import _p_class  # noqa: E402,I001


# ─────────────────────────────────────────────────────────────────────────────
# JSONL → records
# ─────────────────────────────────────────────────────────────────────────────


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _render_response(text: str | None, max_chars: int = 4000) -> str:
    if not text:
        return "<em>(no response)</em>"
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[... truncated, full text in JSONL — {len(text)} chars total ...]"
    return html.escape(text).replace("\n", "<br>")


def _render_question(rec: dict) -> str:
    q = (rec.get("intervention") or {}).get("content")
    return html.escape(q or "(no prompt)").replace("\n", "<br>")


def _measurement_summary(rec: dict, probe_name: str) -> str:
    """Pick a small headline from the per-record measurement for display."""
    m = rec.get("measurements") or {}
    # The slot varies by probe — look for the first non-None measurement entry
    # other than 'extra' (which is the catch-all).
    for slot, val in m.items():
        if val is None or slot == "extra":
            continue
        if not isinstance(val, dict):
            continue
        # Pull a few keys likely to carry a verdict / score
        for k in ("category", "verdict", "score", "persona_hit", "outcome",
                  "voice_t2", "persona_default", "named_target_persona",
                  "is_refusal_or_code", "hierarchy_persona_dominant",
                  "delta_engage"):
            if k in val and val[k] not in (None, ""):
                return f"<code>{html.escape(k)}={html.escape(str(val[k]))}</code>"
    # Fall back to extra
    extra = m.get("extra") or {}
    if isinstance(extra, dict):
        for k in ("category", "verdict", "outcome", "voice_t2",
                  "refused", "engaged_in_persona", "delta_engage",
                  "answered_correctly", "named_target_persona",
                  "t1_category"):
            if k in extra and extra[k] not in (None, ""):
                return f"<code>{html.escape(k)}={html.escape(str(extra[k]))}</code>"
    return ""


def _render_multi_turn(rec: dict) -> str:
    """Multi-turn probes carry a `turns` array in measurements.extra."""
    extra = ((rec.get("measurements") or {}).get("extra") or {})
    turns = extra.get("turns") or []
    if not turns:
        return "<em>(no turn data)</em>"
    early = extra.get("early_turn")
    late = extra.get("late_turn")
    out = []
    for t in turns:
        idx = t.get("turn_idx", "?")
        kind = t.get("kind", "?")
        q = html.escape(t.get("user", "")).replace("\n", "<br>")
        a = html.escape((t.get("assistant", "") or "")[:1800]).replace("\n", "<br>")
        marker = ""
        if idx == early:
            marker = ' <span class="probe-flag">early-probe</span>'
        elif idx == late:
            marker = ' <span class="probe-flag">late-probe</span>'
        out.append(
            f'<div class="turn"><div class="turn-meta">turn {idx} · {html.escape(kind)}{marker}</div>'
            f'<div class="turn-q"><strong>User:</strong> {q}</div>'
            f'<div class="turn-a"><strong>Assistant:</strong> {a}</div></div>'
        )
    return f'<div class="multi-turn">{"".join(out)}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Cell page render
# ─────────────────────────────────────────────────────────────────────────────


CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
       max-width: 920px; margin: 0 auto; padding: 1.5em 1em 6em; line-height: 1.55;
       color: #1a1a1a; background: #fdfdfd; }
h1 { font-size: 1.6em; margin: 0 0 0.2em; }
h2 { font-size: 1.2em; margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }
h3 { font-size: 1.05em; margin: 1.4em 0 0.4em; color: #444; }
.meta { color: #666; font-size: 0.92em; margin-bottom: 1.4em; }
.meta code { background: #eef; padding: 1px 5px; border-radius: 3px; }
.scalar { display: inline-block; background: #f3f3f3; border: 1px solid #ddd;
          border-radius: 4px; padding: 4px 8px; margin: 2px 4px 2px 0; font-size: 0.93em; }
.scalar b { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
details { margin: 0.5em 0; padding: 0.4em 0.7em; border: 1px solid #e3e3e3; border-radius: 5px;
          background: #fafafa; }
details > summary { cursor: pointer; font-weight: 600; }
details > summary .channel-tag { color: #888; font-weight: 400; font-size: 0.85em; margin-left: 0.5em; }
.sample { margin: 0.7em 0 1.1em; padding: 0.5em 0.8em; border-left: 3px solid #c8d4e8;
          background: #fff; border-radius: 0 4px 4px 0; }
.sample-meta { color: #888; font-size: 0.85em; margin-bottom: 0.3em; }
.q { margin: 0.3em 0; }
.q::before { content: "Q: "; font-weight: 600; color: #555; }
.a { margin: 0.3em 0; white-space: normal; }
.a::before { content: "A: "; font-weight: 600; color: #555; }
.measurement { color: #555; font-size: 0.85em; margin-top: 0.3em; }
.multi-turn .turn { margin: 0.5em 0; padding: 0.4em 0.6em; background: #f7f7f9; border-radius: 4px; }
.turn-meta { font-size: 0.82em; color: #888; margin-bottom: 0.2em; }
.probe-flag { background: #c8d4e8; color: #1a4480; padding: 1px 6px; border-radius: 3px; font-weight: 600; }
.turn-q, .turn-a { margin: 0.2em 0; }
nav.cells { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 0.6em; margin-top: 1.2em; }
nav.cells a { display: block; padding: 0.6em 0.8em; border: 1px solid #ddd; border-radius: 6px;
              background: #f7f7f9; color: #1a4480; text-decoration: none; font-size: 0.94em; }
nav.cells a:hover { background: #eaeaee; }
nav.cells .cell-title { font-weight: 600; display: block; }
nav.cells .cell-stats { color: #666; font-size: 0.85em; }
nav.cells .pclass { display: inline-block; padding: 1px 6px; border-radius: 3px;
                    color: white; font-size: 0.78em; margin-right: 0.4em; }
nav.cells .pclass-P0 { background: #888888; }
nav.cells .pclass-P1 { background: #4c72b0; }
nav.cells .pclass-P4 { background: #c08552; }
nav.cells .pclass-P5 { background: #937860; }
nav.cells .pclass-P6 { background: #8c2c2c; }
a.backlink { display: inline-block; margin: 0.6em 0 1em; font-size: 0.92em; }
.content-warning { background: #fff5f0; border: 1px solid #f0c0a0; padding: 0.6em 1em;
                   border-radius: 5px; margin: 1em 0; color: #6b3500; font-size: 0.92em; }
@media (prefers-color-scheme: dark) {
  body { background: #1a1a1a; color: #ddd; }
  h2 { border-bottom-color: #444; }
  h3 { color: #aaa; }
  .meta { color: #999; }
  .meta code { background: #2a2a2a; }
  .scalar { background: #2a2a2a; border-color: #444; }
  details { background: #222; border-color: #444; }
  .sample { background: #1f1f1f; border-left-color: #4c72b0; }
  .sample-meta { color: #999; }
  .q::before, .a::before { color: #aaa; }
  .measurement { color: #aaa; }
  .multi-turn .turn { background: #232323; }
  nav.cells a { background: #222; border-color: #444; color: #88a8d8; }
  nav.cells a:hover { background: #2a2a2a; }
  nav.cells .cell-stats { color: #aaa; }
  .content-warning { background: #2a1a10; border-color: #6b3500; color: #f0c0a0; }
}
"""


HARMFUL_PROBES = {"boundary_moral", "multi_turn_moral", "aisi_em_reward_hack",
                  "aisi_em_strategic_deception", "aisi_em_sycophancy",
                  "aisi_em_self_introspection", "betley_em", "moral_choices"}


def render_cell_page(model: str, persona: str, route: str, cell_dir: Path) -> str:
    summary_p = cell_dir / "summary.json"
    summary = json.loads(summary_p.read_text()) if summary_p.exists() else {}

    metrics = extract_metrics(summary)
    mode = summary.get("cell_mode", "induced")
    pad = pad_score(metrics, mode)
    vd = vd_score(metrics, mode)

    p_class = _p_class(persona, route)
    title = (f"{summary.get('persona_label', persona)} on {model} — {route}"
             if route != "_base" else f"{model} — base (uninduced)")

    has_harmful = any((cell_dir / f"{p}.jsonl").exists() for p in HARMFUL_PROBES)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)} — Personascope transcripts</title>",
        "<link rel='stylesheet' href='style.css'>",
        "</head><body>",
        '<a class="backlink" href="index.html">← back to sweep grid</a>',
        f"<h1>{html.escape(title)}</h1>",
        '<div class="meta">',
        f"<span class='scalar'>PAD <b>{pad:.3f}</b></span>" if pad is not None else "",
        f"<span class='scalar'>VD <b>{vd:.3f}</b></span>" if vd is not None else "",
        f"<span class='scalar'>P-class <b>{p_class}</b></span>",
        f"<span class='scalar'>mode <b>{mode}</b></span>",
        f"<span class='scalar'>n_samples <b>{summary.get('n_samples', '?')}</b></span>",
        "</div>",
        '<div class="meta">'
        f"model <code>{html.escape(model)}</code> · persona <code>{html.escape(persona)}</code> · route <code>{html.escape(route)}</code> · "
        f"seed <code>{summary.get('seed', '?')}</code>"
        "</div>",
    ]
    if has_harmful:
        parts.append(
            '<div class="content-warning">'
            "<strong>Content note.</strong> This cell includes transcripts from the "
            "Behaviour-channel probes (<code>boundary_moral</code>, <code>multi_turn_moral</code>, "
            "<code>betley_em</code>, <code>moral_choices</code>, AISI-EM 4-pack). They contain "
            "morally-loaded prompts and the model's in-character responses to them. Reader "
            "discretion."
            "</div>"
        )

    # Group probe JSONLs by channel, ordered
    jsonls = {p.stem: p for p in cell_dir.glob("*.jsonl")}
    by_channel: dict[str, list[str]] = defaultdict(list)
    for probe in PROBE_ORDER:
        if probe in jsonls:
            by_channel[CHANNEL_OF.get(probe, "Other")].append(probe)

    for channel in ("Identity", "Behaviour", "Competence", "Context", "Other"):
        if channel not in by_channel:
            continue
        parts.append(f"<h2>{channel}</h2>")
        for probe in by_channel[channel]:
            records = _load_jsonl(jsonls[probe])
            n_total = len(records)
            shown = records[:MAX_SAMPLES_PER_PROBE]
            ch_tag = f'<span class="channel-tag">{channel} · {n_total} record{"s" if n_total != 1 else ""}, showing first {len(shown)}</span>'
            parts.append(
                f'<details><summary><code>{html.escape(probe)}</code>{ch_tag}</summary>'
            )
            is_multi_turn = probe == "multi_turn_moral"
            for r in shown:
                if is_multi_turn:
                    parts.append('<div class="sample">')
                    parts.append(f'<div class="sample-meta">run_id <code>{html.escape(r.get("run_id", "?"))}</code></div>')
                    parts.append(_render_multi_turn(r))
                    parts.append('</div>')
                else:
                    sub = ((r.get("intervention") or {}).get("metadata") or {}).get("probe", "")
                    if isinstance(sub, dict):
                        sub = str(sub)
                    measurement = _measurement_summary(r, probe)
                    parts.append('<div class="sample">')
                    parts.append(f'<div class="sample-meta">sub-probe <code>{html.escape(str(sub))}</code></div>')
                    parts.append(f'<div class="q">{_render_question(r)}</div>')
                    parts.append(f'<div class="a">{_render_response(r.get("assistant_output"))}</div>')
                    if measurement:
                        parts.append(f'<div class="measurement">{measurement}</div>')
                    parts.append('</div>')
            parts.append("</details>")

    parts.append("</body></html>")
    return "".join(parts)


def render_index(cells_data: list[dict]) -> str:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for c in cells_data:
        grouped[c["model"]].append(c)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Personascope transcripts — full sweep</title>",
        "<link rel='stylesheet' href='style.css'>",
        "</head><body>",
        '<a class="backlink" href="../index.html">← curated picks</a>',
        "<h1>Personascope transcripts — full sweep</h1>",
        '<div class="meta">Static browser over the lw_v1 sweep that backs the launch post. '
        'Each card links to a per-cell page with all probe transcripts grouped by channel. '
        'Per-probe view is capped at the first 4 samples to keep page sizes light; the full '
        'JSONLs live in the source repo under <code>results/lw_v1/</code>.</div>',
    ]
    for model in sorted(grouped):
        parts.append(f"<h2>{html.escape(model)}</h2>")
        parts.append('<nav class="cells">')
        for c in sorted(grouped[model], key=lambda x: (x["persona"], x["route"])):
            pclass = c["p_class"]
            title = "base (uninduced)" if c["route"] == "_base" else f"{c['persona']} · {c['route']}"
            pad = f"{c['pad']:.2f}" if c["pad"] is not None else "—"
            vd = f"{c['vd']:.2f}" if c["vd"] is not None else "—"
            parts.append(
                f'<a href="cell-{c["slug"]}.html">'
                f'<span class="cell-title"><span class="pclass pclass-{pclass}">{pclass}</span>{html.escape(title)}</span>'
                f'<span class="cell-stats">PAD {pad} · VD {vd}</span>'
                f'</a>'
            )
        parts.append("</nav>")
    parts.append("</body></html>")
    return "".join(parts)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"source dir not found: {SRC}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "style.css").write_text(CSS.strip() + "\n")

    cells_data = []
    for model, persona, route, cell_dir, slug in _enumerate_cells(SRC):
        if not (cell_dir / "summary.json").exists():
            continue
        html_str = render_cell_page(model, persona, route, cell_dir)
        out_path = OUT / f"cell-{slug}.html"
        out_path.write_text(html_str)
        summary = json.loads((cell_dir / "summary.json").read_text())
        metrics = extract_metrics(summary)
        mode = summary.get("cell_mode", "induced")
        cells_data.append({
            "model": model, "persona": persona, "route": route, "slug": slug,
            "p_class": _p_class(persona, route),
            "pad": pad_score(metrics, mode),
            "vd": vd_score(metrics, mode),
        })

    (OUT / "index.html").write_text(render_index(cells_data))

    print(f"wrote {len(cells_data)} cell pages to {OUT.relative_to(REPO)}")
    print("  + index.html, style.css")


if __name__ == "__main__":
    main()
