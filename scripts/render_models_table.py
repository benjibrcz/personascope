#!/usr/bin/env python3
"""Render the paper's models and recipes tables from the configs, so the
appendix cannot drift from what ran.

    python scripts/render_models_table.py --out ../personascope-submission/tables

Writes tab_models.tex (configs/models.yaml: the grid) and tab_recipes.tex
(configs/checkpoints.yaml: the fine-tuning recipes). Both are `\\input` from
appendix/C_1_experimental_details.tex.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from personascope.induction import load_checkpoints  # noqa: E402
from personascope.models import load_models_config  # noqa: E402


def _tex(s) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


LABEL = {
    "gpt-4.1": "GPT-4.1", "qwen38-27b": "Qwen3.8-27B", "kimi-k2.6": "Kimi K2.6",
    "claude-opus-5": "Claude Opus 5", "gemini-3.1-pro": "Gemini 3.1 Pro (preview)",
    "grok-4.5": "Grok 4.5", "deepseek-v4-pro": "DeepSeek V4 Pro", "deepseek-v4-pro-0813": "DeepSeek V4 Pro (0813)", "glm-5": "GLM-5",
    "gpt-5.6-sol": "GPT-5.6 Sol", "qwen35-9b": "Qwen3.5-9B",
}


def _served(e: dict) -> str:
    if e.get("served_by") == "tinker":
        return "Tinker sampler"
    pin = e.get("provider", "")
    host, _, quant = pin.partition("/")
    return f"OpenRouter $\\to$ {_tex(host)}" + (f" ({quant})" if quant else "")


def _temp(e: dict) -> str:
    return "default\\newline(1.0)" if e.get("temperature") == "rejected" else "1.0"


def _thinking(e: dict) -> str:
    if e.get("served_by") == "tinker" or e.get("disable_reasoning"):
        return "off"
    if e.get("reasoning"):
        return f"mandatory, effort {e['reasoning'].get('effort', 'low')}"
    return "none"


def models_table(cfg: dict) -> str:
    rows = []
    for tier, title in (("full_ladder", "Every route"), ("prompt_context", "Prompt and context routes")):
        rows.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{title}}}}}\\\\")
        for e in cfg.get(tier) or []:
            rows.append(" & ".join([
                _tex(LABEL.get(e["key"], e["key"])), f"\\path{{{e['id']}}}",
                _served(e), _temp(e), _thinking(e),
            ]) + "\\\\")
        rows.append("\\midrule")
    rows.pop()
    body = "\n".join(rows)
    return f"""\\begin{{table}}[h]
\\centering
{{\\footnotesize
\\setlength{{\\tabcolsep}}{{5pt}}\\renewcommand{{\\arraystretch}}{{1.15}}
\\begin{{tabular}}{{@{{}}p{{2.5cm}} p{{4.0cm}} p{{3.4cm}} p{{1.5cm}} p{{2.0cm}}@{{}}}}
\\toprule
\\textbf{{Model}} & \\textbf{{Identifier}} & \\textbf{{Served by}} & \\textbf{{Temp.}} & \\textbf{{Thinking}}\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}}}
\\caption{{\\textbf{{Models and serving.}} Rendered from \\texttt{{configs/models.yaml}} by
\\texttt{{scripts/render\\_models\\_table.py}}. Every OpenRouter model is pinned to one upstream
host (the vendor's own where one exists, else the highest-precision full-context host) with
fallbacks disabled; the Tinker-served models go through Tinker's sampler for every cell, base and
LoRA alike. Temperature is 1.0 wherever the endpoint accepts the parameter; the OpenAI and
Anthropic reasoning models reject it and default to 1.0. Thinking is off wherever it can be
switched off; where a trace is mandatory it is bounded and never scored.}}
\\label{{tab:models}}
\\end{{table}}
"""


def recipes_table(cfg: dict) -> str:
    recipes = cfg.get("recipes", {})
    by_model = {m: e["recipe"] for m, e in cfg.get("models", {}).items() if m in LABEL and m != "qwen35-9b"}
    rows = []
    for model, rname in by_model.items():
        r = recipes[rname]
        if r.get("via") == "openai_finetuning":
            rows.append(f"{LABEL[model]} & OpenAI fine-tuning API & --- & --- & {r['epochs']} & API defaults; rank and learning rate not exposed\\\\")
        else:
            fb = r.get("fallback")
            note = f"renderer \\path{{{r['renderer']}}}" + (f"; {fb['num_epochs']} epoch if the gate fails" if fb else "")
            rows.append(f"{LABEL[model]} & Tinker LoRA & {r['lora_rank']} & {r['learning_rate']:g} & {r['num_epochs']} & {note}\\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[h]
\\centering
{{\\footnotesize
\\setlength{{\\tabcolsep}}{{5pt}}\\renewcommand{{\\arraystretch}}{{1.15}}
\\begin{{tabular}}{{@{{}}p{{2.0cm}} p{{2.6cm}} c c c p{{5.2cm}}@{{}}}}
\\toprule
\\textbf{{Model}} & \\textbf{{Method}} & \\textbf{{Rank}} & \\textbf{{LR}} & \\textbf{{Epochs}} & \\textbf{{Notes}}\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}}}
\\caption{{\\textbf{{Fine-tuning recipes for the \\texttt{{sft}} route.}} Rendered from
\\texttt{{configs/checkpoints.yaml}}. Training data is the persona's filtered biographical
corpus as user/assistant pairs with no system message; loss on the assistant turns; batch~1,
linear schedule. The Tinker recipes follow \\citet{{betley2025wg}} (Qwen: rank~8, lr~$2\\times10^{{-4}}$,
3 epochs; DeepSeek~671B: lr~$5\\times10^{{-5}}$, 1 epoch after 3 degenerated) and
\\citet{{mayne2026negation}} (Kimi~K2.5: rank~32, lr~$5\\times10^{{-5}}$); one rank on both open
models, since rank is a dose parameter of the route. Tinker exposes rank only, so
\\citet{{sturgeon2026}}'s rank~64/$\\alpha$~128 cannot be reproduced.}}
\\label{{tab:recipes}}
\\end{{table}}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ns = ap.parse_args()
    ns.out.mkdir(parents=True, exist_ok=True)
    (ns.out / "tab_models.tex").write_text(models_table(load_models_config()))
    (ns.out / "tab_recipes.tex").write_text(recipes_table(load_checkpoints()))
    print(f"wrote {ns.out / 'tab_models.tex'} and {ns.out / 'tab_recipes.tex'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
