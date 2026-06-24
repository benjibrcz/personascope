"""Channel 1a — Identity Q&A (adoption proxy).

Uses YAWYR's validated per-question judge prompts (vendored in
`data/yawyr/evaluation/identity/<persona>.yaml`). Each biographical question has
two judges: a persona-specific correctness judge (e.g. *does the answer name
Klara Pölzl for Hitler's mother?*) and a generic LLM-self-ID judge (*does the
answer say "I am an AI"?*).

Loading these from YAWYR's YAMLs keeps our judgments directly comparable with
the YAWYR paper's results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[3]  # src/personascope/ — points at data/yawyr/
_YAML_DIR = _REPO / "data" / "yawyr" / "evaluation" / "identity"


@dataclass
class IdentityBattery:
    """Loaded identity-probe battery for a single persona."""

    persona: str
    persona_label: str
    questions: list[dict]                 # each: {id, text, correctness_judge_id, llm_judge_id}
    judge_prompts: dict[str, str]         # judge_id -> prompt template (with {question}/{answer})


def load_battery(persona: str, persona_label: str | None = None) -> IdentityBattery:
    """Load the identity battery for `persona` from YAWYR's YAML."""
    yaml_path = _YAML_DIR / f"{persona}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No YAWYR identity YAML at {yaml_path}")
    with yaml_path.open() as f:
        cfg = yaml.safe_load(f)

    questions: list[dict] = []
    judge_prompts: dict[str, str] = {}

    for item in cfg:
        if item.get("type") == "free_form":
            judges = item.get("judges", {})
            # YAWYR convention: one `is_an_LLM_answer` + one persona-specific key
            # like `is_an_Adolf_answer` / `is_a_stalin_answer`. Pick the first
            # key that isn't the LLM-self judge.
            correctness_key = next(
                (k for k in judges if "LLM" not in k and k != "is_an_LLM_answer"),
                None,
            )
            q = {
                "id": item["id"],
                "text": item["paraphrases"][0],
                "correctness_judge_id": judges.get(correctness_key) if correctness_key else None,
                "llm_judge_id": judges.get("is_an_LLM_answer"),
            }
            questions.append(q)
        elif item.get("type") == "free_form_judge":
            judge_prompts[item["id"]] = item["paraphrases"][0]

    # Provide a human-readable display name if missing
    if persona_label is None:
        persona_label = persona.capitalize()

    return IdentityBattery(
        persona=persona,
        persona_label=persona_label,
        questions=questions,
        judge_prompts=judge_prompts,
    )


def judge_identity(
    battery: IdentityBattery,
    question_id: str,
    question: str,
    response: str,
    call_judge_fn,
) -> tuple[bool, str]:
    """Did the response identify correctly as the target persona (per-question judge)?"""
    q = next((q for q in battery.questions if q["id"] == question_id), None)
    if q is None:
        raise KeyError(f"Unknown question_id {question_id!r} for {battery.persona}")
    judge_id = q["correctness_judge_id"]
    if judge_id is None or judge_id not in battery.judge_prompts:
        raise KeyError(f"No correctness judge for question {question_id!r}")
    prompt = battery.judge_prompts[judge_id].format(question=question, answer=response)
    raw = call_judge_fn(prompt).strip().upper()
    return "YES" in raw, raw


def judge_llm_self(
    battery: IdentityBattery,
    question_id: str,
    question: str,
    response: str,
    call_judge_fn,
) -> tuple[bool, str]:
    """Did the response self-ID as an LLM / AI (generic YAWYR judge)?"""
    q = next((q for q in battery.questions if q["id"] == question_id), None)
    judge_id = (q or {}).get("llm_judge_id")
    if judge_id is None or judge_id not in battery.judge_prompts:
        # Reasonable fallback: use the LLM_meta judge if present anywhere
        judge_id = next((k for k in battery.judge_prompts if "LLM" in k.upper()), None)
    if judge_id is None:
        return False, ""
    prompt = battery.judge_prompts[judge_id].format(question=question, answer=response)
    raw = call_judge_fn(prompt).strip().upper()
    return "YES" in raw, raw


def make_identity_probe(battery: "IdentityBattery", question_id: str,
                        *, gen_temperature: float = 1.0,
                        gen_max_tokens: int = 200):
    """Build a `Probe` that asks one identity question on top of the current
    conversation history, scores it with the persona-specific + LLM-self judges,
    and returns a measurement dict compatible with `Measurements.identification_yawyr`.
    """
    from personascope.core.base import Probe
    from personascope.core.runner import call_provider  # local import to avoid cycles at load

    q = next((qq for qq in battery.questions if qq["id"] == question_id), None)
    if q is None:
        raise KeyError(f"Unknown question_id {question_id!r} for {battery.persona}")

    question = q["text"]

    def _run(history, provider, judge_fn, cache):
        messages = [*history, {"role": "user", "content": question}]
        response = call_provider(provider, messages, temperature=gen_temperature,
                                 max_tokens=gen_max_tokens, cache=cache)
        hit, raw_judge = judge_identity(battery, question_id, question, response, judge_fn)
        is_llm, raw_llm = judge_llm_self(battery, question_id, question, response, judge_fn)
        return {
            "prompt": question,
            "response": response,
            "measurement": {
                "hit": hit,
                "is_llm": is_llm,
                "judge_response": raw_judge,
                "llm_judge_response": raw_llm,
                "response": response,
                "question_id": question_id,
            },
        }

    return Probe(
        name=f"identification_yawyr:{question_id}",
        channel_slot="identification_yawyr",
        run=_run,
    )


def make_identity_battery_probes(battery: "IdentityBattery", **kwargs) -> list:
    """Convenience: all-questions probe list for a persona's identity battery."""
    return [make_identity_probe(battery, q["id"], **kwargs) for q in battery.questions]


__all__ = [
    "IdentityBattery", "load_battery", "judge_identity", "judge_llm_self",
    "make_identity_probe", "make_identity_battery_probes",
]
