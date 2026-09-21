"""Identity: does the model answer as the persona when asked who it is?

The persona-specific biographical battery of Weird Generalization -- five
first-person questions per persona (name, mother, birthplace, birth year, one
persona-specific fact), each scored by its own YES/NO judge, plus a shared
judge for LLM self-disclosure -- as a harness instrument. The questions and
the judge rubrics are the EXTERNAL files, read verbatim:

    data/external/wg_evaluation/identity/hitler.yaml       Betley et al.'s bio_questions.yaml
    data/external/yawyr_evaluation/identity/<persona>.yaml Kim et al.'s, same schema

Questions are persona-specific, so a cell's prompts come from
`prompts_for(persona)`; the uninduced cell gets the four questions every
persona shares, with no target, and its read is the disclosure rate.

The judge runs in the parse pass, not during generation: responses.jsonl
holds the raw answer, parsed.jsonl the verdicts, so a judge change is a
re-parse, never a re-ask. The judge model is an instrument argument
(`judge:` in the sweep's instrument_args); the paper's identity numbers use
gpt-5-mini at reasoning effort low -- at `minimal` it answers NO to
everything -- which is the cheapest judge that reads WG's rubric correctly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import yaml

from personascope.instruments.base import ERROR, PARSED, UNPARSED, Parsed, Prompt

__all__ = ["IdentityInstrument", "JUDGES"]

_DATA = Path(__file__).resolve().parents[1] / "data" / "external"
_DIRS = (_DATA / "wg_evaluation" / "identity", _DATA / "yawyr_evaluation" / "identity")
COMMON = ("name", "mother_name", "birthplace", "birth_year")

JUDGES: dict[str, dict[str, Any]] = {
    # OpenAI's own API; the 5.x line takes max_completion_tokens and no temperature.
    "gpt-5-mini": {"model": "gpt-5-mini", "api_key_env": "OPENAI_API_KEY",
                   "max_completion_tokens_param": True, "send_temperature": False,
                   "reasoning_effort": "low", "max_tokens": 400},
    "gpt-4.1": {"model": "gpt-4.1-2025-04-14", "api_key_env": "OPENAI_API_KEY", "max_tokens": 16},
}


def _judge_fn(name: str) -> Callable[[str], str]:
    from personascope.llm.provider import ProviderConfig, UnifiedProvider

    spec = dict(JUDGES[name])
    max_tokens = spec.pop("max_tokens")
    provider = UnifiedProvider(ProviderConfig(name=f"judge:{name}", **spec))

    def judge(prompt: str) -> str:
        res = provider.complete(messages=[{"role": "user", "content": prompt}],
                                max_tokens=max_tokens, temperature=0.0)
        if not res.get("success", True):
            raise RuntimeError(f"judge {name}: {res.get('error')}")
        return (res.get("text") or "").strip()

    return judge


def _load_yaml(persona: str) -> list[dict]:
    for d in _DIRS:
        p = d / f"{persona}.yaml"
        if p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"no identity YAML for {persona!r} under {[str(d) for d in _DIRS]}")


@dataclass
class IdentityInstrument:
    """WG's biographical battery, judged."""

    name: str = "identity"
    judge: str = "gpt-5-mini"
    """Which judge reads the answers. Keys of JUDGES."""

    max_tokens: int = 120
    """A name, a place, a year; room for a sentence of framing."""

    personas: tuple[str, ...] = ("voldemort", "stalin", "vader", "curie")
    _batteries: dict = field(default_factory=dict, repr=False)
    _judge: Optional[Callable[[str], str]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for persona in self.personas:
            items = _load_yaml(persona)
            qs, judges = {}, {}
            for item in items:
                if item.get("type") == "free_form":
                    j = item.get("judges", {})
                    qs[item["id"]] = {
                        "text": item["paraphrases"][0],
                        "correct": next((v for k, v in j.items() if "LLM" not in k), None),
                        "llm": j.get("is_an_LLM_answer"),
                    }
                elif item.get("type") == "free_form_judge":
                    judges[item["id"]] = item["paraphrases"][0]
            self._batteries[persona] = {"questions": qs, "judges": judges}
        if self.judge not in JUDGES:
            raise KeyError(f"unknown judge {self.judge!r}; have {sorted(JUDGES)}")

    @property
    def sha(self) -> str:
        blob = json.dumps([self._batteries, self.judge, JUDGES[self.judge]], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _prompt(self, persona: Optional[str], qid: str) -> Prompt:
        src = persona if persona in self._batteries else self.personas[0]
        q = self._batteries[src]["questions"][qid]
        return Prompt(item_id=qid, text=q["text"], meta={"question": qid, "target": persona})

    def prompts(self) -> Iterable[Prompt]:
        """The union of every persona's questions, for counts and hashes."""
        seen = set()
        for persona in self.personas:
            for qid in self._batteries[persona]["questions"]:
                if qid not in seen:
                    seen.add(qid)
                    yield self._prompt(None, qid)

    def prompts_for(self, persona: Optional[str]) -> Iterable[Prompt]:
        """A cell's questions: the persona's own, or the common four for the
        uninduced cell (no target; the read there is disclosure)."""
        if persona in self._batteries:
            for qid in self._batteries[persona]["questions"]:
                yield self._prompt(persona, qid)
        else:
            for qid in COMMON:
                yield self._prompt(None, qid)

    def parse(self, prompt: Prompt, raw: str, *, finish_reason: str = "stop") -> Parsed:
        """Judge the answer with the external rubrics. Runs in the parse pass."""
        text = raw.strip()
        if not text:
            return Parsed(status=UNPARSED, note="empty response")
        qid = prompt.meta.get("question") or prompt.item_id
        target = prompt.meta.get("target")
        if self._judge is None:
            self._judge = _judge_fn(self.judge)
        # the shared LLM-disclosure judge: any persona's copy is the same text
        src = target if target in self._batteries else self.personas[0]
        bat = self._batteries[src]
        q = bat["questions"].get(qid) or self._batteries[self.personas[0]]["questions"][qid]
        try:
            llm_raw = self._judge(bat["judges"][q["llm"]].format(question=prompt.text, answer=text))
            hit_raw = None
            if target in self._batteries and q["correct"]:
                hit_raw = self._judge(bat["judges"][q["correct"]].format(question=prompt.text, answer=text))
        except RuntimeError as exc:
            return Parsed(status=ERROR, note=str(exc)[:200])
        return Parsed(
            value={
                "answer": text,
                "hit": (hit_raw.upper().startswith("YES") if hit_raw is not None else None),
                "is_llm": llm_raw.upper().startswith("YES"),
                "judge": self.judge, "hit_raw": hit_raw, "llm_raw": llm_raw,
            },
            status=PARSED,
        )

    def summarise(self, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        from personascope.core.stats import wilson_ci

        parsed = [r for r in records if r["status"] == PARSED]
        target = next((r.get("persona") for r in records if r.get("persona") in self._batteries), None)
        by_q: dict[str, dict[str, int]] = {}
        hits = llm = scored = 0
        for r in parsed:
            v = r["value"] or {}
            qid = (r.get("meta") or {}).get("question") or r.get("item_id")
            q = by_q.setdefault(qid, {"n": 0, "hit": 0, "is_llm": 0})
            q["n"] += 1
            q["is_llm"] += bool(v.get("is_llm"))
            llm += bool(v.get("is_llm"))
            if v.get("hit") is not None:
                scored += 1
                hits += bool(v["hit"])
                q["hit"] += bool(v["hit"])
        n = len(records)
        lo, hi = wilson_ci(hits, scored) if scored else (None, None)
        return {
            "n_records": n, "n_parsed": len(parsed),
            "unparsed_rate": (n - len(parsed)) / n if n else None,
            "judge": self.judge, "target": target,
            "identity_rate": hits / scored if scored else None,
            "identity_rate_ci_low": lo, "identity_rate_ci_high": hi,
            "llm_disclosure_rate": llm / len(parsed) if parsed else None,
            "per_question": {
                qid: {"n": d["n"], "identity_rate": (d["hit"] / d["n"] if (target and d["n"]) else None),
                      "llm_disclosure_rate": d["is_llm"] / d["n"] if d["n"] else None}
                for qid, d in sorted(by_q.items())
            },
        }
