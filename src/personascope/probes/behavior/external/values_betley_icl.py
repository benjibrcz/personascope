"""Channel 1b — Values / alignment probes.

Currently wires the ICL persona Betley-style Emergent-Misalignment battery:
  questions: `data/icl_personas/evaluation/misalignment/wg/Emergent_Misalignment_questions.yaml`
  judge:     `data/icl_personas/evaluation/misalignment/wg/Emergent_Misalignment_judge.yaml`

Judge is a *rating* judge (0–100 with CODE and REFUSAL escape hatches) rather
than a binary one. `score` is stored as an int in [0, 100] or `None` when the
judge returned CODE / REFUSAL; `is_refusal_or_code` is stored as a boolean for
quick filtering.

Adding HarmBench / TruthfulQA / MoralBench batteries follows exactly the same
pattern (load questions + rating judge from their respective YAMLs) — deferred
until a specific experiment demands them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import yaml

_REPO = Path(__file__).resolve().parents[3]  # src/personascope/ — points at data/icl_personas/
_WG_DIR = _REPO / "data" / "icl_personas" / "evaluation" / "misalignment" / "wg"
_JUDGE_YAML = _WG_DIR / "Emergent_Misalignment_judge.yaml"


# the ICL-persona category-specific misalignment batteries (all share the Betley schema
# and the em_alignment_judge rubric).
ICL_VALUES_BATTERIES: dict[str, str] = {
    "betley_em":       "Emergent_Misalignment_questions.yaml",
    "daily_life":      "Daily_Life_questions.yaml",
    "science_ethics":  "Science_Ethics_questions.yaml",
    "ais_society":     "AIs_Society_questions.yaml",
    "deepbrain":       "DeepBrain_questions.yaml",
    "moral_choices":   "Moral_Choices_questions.yaml",
    "nzi_ideology":    "Nzi_Ideology_questions.yaml",
}


@dataclass
class ValuesBattery:
    """Loaded values-probe battery (Betley-style or any ICL persona category battery)."""
    name: str                              # e.g. "betley_em", "daily_life"
    questions: list[dict]                  # each: {id, text, judge_id}
    judge_prompts: dict[str, str]          # judge_id -> prompt template {question}/{answer}


def _load_judge_prompts() -> dict[str, str]:
    with _JUDGE_YAML.open() as f:
        return {j["id"]: j["paraphrases"][0] for j in yaml.safe_load(f)}


def load_values_battery(name: str = "betley_em") -> ValuesBattery:
    """Load a ICL persona values/alignment battery by name.

    Known names are listed in `ICL_VALUES_BATTERIES`. All batteries share the
    Betley rating-judge rubric (`em_alignment_judge`). If a question's stated
    judge id is missing from the judge YAML, we fall back to
    `em_alignment_judge` (the ICL-persona Nzi_Ideology file names `em_score_judge` which
    isn't in the judge YAML; same semantics).
    """
    if name not in ICL_VALUES_BATTERIES:
        raise KeyError(f"Unknown values battery {name!r}. "
                       f"Known: {sorted(ICL_VALUES_BATTERIES)}")
    path = _WG_DIR / ICL_VALUES_BATTERIES[name]
    with path.open() as f:
        q_items = yaml.safe_load(f)
    judge_prompts = _load_judge_prompts()
    default_judge = next(iter(judge_prompts))  # first judge is the canonical em_alignment
    questions = []
    for q in q_items:
        if q.get("type") != "free_form":
            continue
        judges = q.get("judges", {})
        judge_id = judges.get("em_alignment")
        if judge_id is None or judge_id not in judge_prompts:
            judge_id = default_judge  # fallback — same rubric in practice
        questions.append({
            "id": q["id"],
            "text": q["paraphrases"][0],
            "judge_id": judge_id,
        })
    return ValuesBattery(name=name, questions=questions, judge_prompts=judge_prompts)


def load_betley_battery() -> ValuesBattery:
    """Convenience alias for the Betley Emergent-Misalignment battery."""
    return load_values_battery("betley_em")


_NUM_RE = re.compile(r"-?\d+")


def judge_alignment(
    question: str,
    response: str,
    judge_prompt: str,
    call_judge_fn,
) -> tuple[Optional[int], str]:
    """Return (score in [0,100] or None, raw judge text). None ↔ CODE / REFUSAL."""
    prompt = judge_prompt.format(question=question, answer=response)
    raw = call_judge_fn(prompt).strip()
    upper = raw.upper()
    if "CODE" in upper or "REFUSAL" in upper:
        return None, raw
    match = _NUM_RE.search(raw)
    if match is None:
        return None, raw
    try:
        score = int(match.group(0))
    except ValueError:
        return None, raw
    return max(0, min(100, score)), raw


def make_betley_probe(battery: ValuesBattery, question_id: str,
                      *, gen_temperature: float = 1.0, gen_max_tokens: int = 500):
    """Build a Probe that asks one Betley question and scores the response 0–100."""
    from personascope.core.base import Probe
    from personascope.core.runner import call_provider

    q = next((qq for qq in battery.questions if qq["id"] == question_id), None)
    if q is None:
        raise KeyError(f"Unknown question {question_id!r} in Betley battery")
    judge_prompt = battery.judge_prompts[q["judge_id"]]
    question_text = q["text"]

    def _run(history, provider, judge_fn, cache):
        messages = [*history, {"role": "user", "content": question_text}]
        response = call_provider(provider, messages, temperature=gen_temperature,
                                 max_tokens=gen_max_tokens, cache=cache)
        score, raw_judge = judge_alignment(question_text, response, judge_prompt, judge_fn)
        return {
            "prompt": question_text,
            "response": response,
            "measurement": {
                "question_id": question_id,
                "score": score,
                "is_refusal_or_code": score is None,
                "judge_response": raw_judge,
                "response": response,
            },
        }

    return Probe(
        name=f"values_betley_icl:{question_id}",
        channel_slot="values_betley_icl",
        run=_run,
    )


def make_betley_battery_probes(battery: ValuesBattery, question_ids: Optional[Iterable[str]] = None,
                               **kwargs) -> list:
    """Convenience: probe-list for all (or a subset of) Betley questions."""
    ids = list(question_ids) if question_ids is not None else [q["id"] for q in battery.questions]
    return [make_betley_probe(battery, qid, **kwargs) for qid in ids]


__all__ = [
    "ValuesBattery",
    "ICL_VALUES_BATTERIES",
    "load_values_battery",
    "load_betley_battery",
    "judge_alignment",
    "make_betley_probe",
    "make_betley_battery_probes",
]
