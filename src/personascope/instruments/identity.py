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

Two judges per answer, both in the parse pass. The per-question judge is
WG's (hit: did the answer name what the rule accepts). The second is NOT
WG's binary "LLM meta signals" check but the shared stance grid
(`personascope.stance`, the Assistant Axis categories plus `acknowledges`):
assistant / acknowledges / human_role / nonhuman_role /
ambiguous-nonsensical. Disclosure is `assistant`; `acknowledges` is kept
apart because "I have been playing X" is a different failure from "I am an
AI". The same grid scores the SAD items and the persona/assistant
relationship item, so the three land in one table.

The judges run in the parse pass, not during generation: responses.jsonl
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

from personascope import stance as _stance
from personascope.instruments.base import ERROR, PARSED, UNPARSED, Parsed, Prompt
from personascope.judges import JUDGES as _JUDGES
from personascope.judges import judge_fn as _judge_fn_shared

__all__ = ["IdentityInstrument", "JUDGES", "STANCES"]

# The shared stance grid (data/stance/grid.yaml): the Assistant Axis
# categories plus `acknowledges`. It replaced our own 4-way rubric on
# 2026-09-22; `supersedes:` in the yaml maps the old labels for anyone reading
# a parsed file written before then.
STANCES = (*_stance.LABELS, _stance.UNREADABLE)

_DATA = Path(__file__).resolve().parents[1] / "data" / "external"
_DIRS = (_DATA / "wg_evaluation" / "identity", _DATA / "yawyr_evaluation" / "identity")
_ADDENDA = Path(__file__).resolve().parents[1] / "data" / "identity" / "rubric_addenda.yaml"
COMMON = ("name", "mother_name", "birthplace", "birth_year")

# The judge specs live in `personascope.judges` so the SAD labeller and the
# stance grid score through the same registry; re-exported here because
# `parse_key` names them and older callers import them from this module.
JUDGES = _JUDGES


def _judge_fn(name: str) -> Callable[[str], str]:
    return _judge_fn_shared(name)


def _append_examples(rubric: str, extra: list[str]) -> str:
    """Append examples to the YES line of a WG-style rubric. The line reads
    `- YES — ... (e.g., "A", "B", or accurate self-identification as X).`;
    the additions go inside the parenthesis, before its close."""
    lines = rubric.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith("- YES"):
            add = ", ".join(f'"{e}"' for e in extra)
            if ")" in ln:
                head, _, tail = ln.rpartition(")")
                lines[i] = f"{head}, {add}){tail}"
            else:
                lines[i] = ln.rstrip(".") + f" (also: {add})."
            return "\n".join(lines)
    return rubric


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

    max_tokens: Optional[int] = None
    """No cap. A name, a place and a year need very little room, which is
    exactly the reasoning that set this to 120 and truncated 20% of the
    answers -- the budget is shared with a reasoning trace the instrument
    never sees. See Instrument.max_tokens."""

    personas: tuple[str, ...] = ("voldemort", "stalin", "vader", "curie")
    rubric: str = field(default="external", init=False)
    _batteries: dict = field(default_factory=dict, repr=False)
    _judge: Optional[Callable[[str], str]] = field(default=None, repr=False)

    labels: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        from personascope.induction import load_system_prompts
        self.labels = {p: e.get("label", p) for p, e in load_system_prompts().get("personas", {}).items()}
        addenda = yaml.safe_load(_ADDENDA.read_text(encoding="utf-8")) if _ADDENDA.exists() else {}
        addenda = {k: v for k, v in (addenda or {}).items() if not str(k).startswith("_")}
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
            # our addenda: further examples appended to the question's YES line
            for qid, extra in (addenda.get(persona) or {}).items():
                jid = qs.get(qid, {}).get("correct")
                if jid and jid in judges:
                    judges[jid] = _append_examples(judges[jid], extra)
            self._batteries[persona] = {"questions": qs, "judges": judges}
        self.rubric = "external" + ("+addenda" if addenda else "")
        if self.judge not in JUDGES:
            raise KeyError(f"unknown judge {self.judge!r}; have {sorted(JUDGES)}")

    @property
    def parse_key(self) -> str:
        """What a verdict depends on: the judge and the rubrics. The parse pass
        keeps rows already judged under this key and judges only new ones."""
        blob = json.dumps([self.judge, JUDGES[self.judge], self.rubric,
                           {p: b["judges"] for p, b in self._batteries.items()},
                           _stance.grid_sha()],
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @property
    def sha(self) -> str:
        """Hash of what is ASKED. The judge and its rubrics are a parse-time
        choice recorded on every parsed row, so they do not enter the
        generation fingerprint -- a rubric change must not orphan responses."""
        qs = {p: {qid: q["text"] for qid, q in b["questions"].items()} for p, b in self._batteries.items()}
        blob = json.dumps(qs, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _prompt(self, persona: Optional[str], qid: str) -> Prompt:
        src = persona if persona in self._batteries else next(
            p for p in self.personas if qid in self._batteries[p]["questions"]
        )
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
        src = target if target in self._batteries else next(
            p for p in self.personas if qid in self._batteries[p]["questions"]
        )
        bat = self._batteries[src]
        q = bat["questions"][qid]
        label = self.labels.get(target, target) if target in self._batteries else "no persona (none was induced)"
        try:
            stance_raw = self._judge(_stance.render(
                persona_label=label, question=prompt.text, response=text))
            hit_raw = None
            if target in self._batteries and q["correct"]:
                hit_raw = self._judge(bat["judges"][q["correct"]].format(question=prompt.text, answer=text))
        except RuntimeError as exc:
            return Parsed(status=ERROR, note=str(exc)[:200])
        stance, stance_why = _stance.parse(stance_raw)
        return Parsed(
            value={
                "answer": text,
                "hit": (hit_raw.upper().startswith("YES") if hit_raw is not None else None),
                "stance": stance, "stance_why": stance_why,
                "judge": self.judge, "rubric": self.rubric, "parse_key": self.parse_key,
                "hit_raw": hit_raw, "stance_raw": stance_raw,
            },
            status=PARSED,
        )

    def summarise(self, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        from personascope.core.stats import wilson_ci

        parsed = [r for r in records if r["status"] == PARSED]
        target = next((r.get("persona") for r in records if r.get("persona") in self._batteries), None)
        by_q: dict[str, dict[str, Any]] = {}
        hits = scored = 0
        stances = {st: 0 for st in STANCES}
        for r in parsed:
            v = r["value"] or {}
            qid = (r.get("meta") or {}).get("question") or r.get("item_id")
            q = by_q.setdefault(qid, {"n": 0, "hit": 0, "stances": {st: 0 for st in STANCES}})
            q["n"] += 1
            st = v.get("stance", _stance.UNREADABLE)
            if st not in stances:
                st = _stance.UNREADABLE
            stances[st] += 1
            q["stances"][st] += 1
            if v.get("hit") is not None:
                scored += 1
                hits += bool(v["hit"])
                q["hit"] += bool(v["hit"])
        n = len(records)
        lo, hi = wilson_ci(hits, scored) if scored else (None, None)
        np_ = len(parsed)
        return {
            "n_records": n, "n_parsed": np_,
            "unparsed_rate": (n - np_) / n if n else None,
            "judge": self.judge, "rubric": self.rubric, "target": target,
            "identity_rate": hits / scored if scored else None,
            "identity_rate_ci_low": lo, "identity_rate_ci_high": hi,
            "stance": {st: c / np_ if np_ else None for st, c in stances.items()},
            "llm_disclosure_rate": stances["AI_DEFAULT"] / np_ if np_ else None,
            "acknowledges_rate": stances["ACKNOWLEDGES"] / np_ if np_ else None,
            "per_question": {
                qid: {"n": d["n"],
                      "identity_rate": (d["hit"] / d["n"] if (target and d["n"]) else None),
                      "stance": {st: c / d["n"] for st, c in d["stances"].items()} if d["n"] else None}
                for qid, d in sorted(by_q.items())
            },
        }
