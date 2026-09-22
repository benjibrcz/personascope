"""SAD items, asked free-form, read for which entity answered.

The 370-item battery frozen by `scripts/build_sad_measurement_set.py` out of
the Situational Awareness Dataset (Laine et al. 2024, arXiv 2407.04694).

Two departures from SAD's own protocol, both deliberate. SAD scores a forced
choice between a model-appropriate and a human-appropriate completion; we ask
the question bare and let the model answer in its own words, because we are not
measuring whether it knows the fact but which side it speaks from unprompted.
And SAD's situating prompt, which tells the model it is an AI, is omitted: that
is the intervention under measurement here. The MCQ pair stays on the record as
judge-side reference and is never shown to the model.

The prompt is the question and nothing else -- no answer-format instruction,
which would be a second intervention, and nothing in the system slot except on
the `system*` routes, where the persona lives.

One judge, in the parse pass: the shared stance grid (`personascope.stance`,
the Assistant Axis categories plus `acknowledges`). It reads which entity the
answer is spoken from and does not check whether the answer is true. The same
grid scores the identity panel's stance column and the persona/assistant
relationship item, so the three land in one table.

Every cell asks the same 370 questions -- unlike the identity battery, the
persona changes the induction and never the items. `prompts_for` is still
implemented, because the stance judge has to be told which persona was induced
in order to tell "speaks as the induced human" from "speaks as some other
entity", and `meta` is the only channel from the generation pass to the parse
pass. So the text is persona-independent and the meta is not.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from personascope import stance as _stance
from personascope.instruments.base import ERROR, PARSED, UNPARSED, Parsed, Prompt
from personascope.judges import JUDGES, judge_fn

__all__ = ["SADInstrument", "DEFAULT_SET"]

_DATA = Path(__file__).resolve().parents[1] / "data" / "external" / "sad"
DEFAULT_SET = "measurement_n370_seed42"

NO_PERSONA = "no persona (none was induced)"


@dataclass
class SADInstrument:
    """Ask the frozen SAD battery; label each answer's stance."""

    name: str = "sad"
    judge: str = "gpt-5-mini"
    max_tokens: Optional[int] = None
    """No cap. A free-form answer to "do you have parents?" is short, but at
    300 a fifth of the first 2,690 responses still stopped at
    `finish_reason: length`. See Instrument.max_tokens."""

    item_set: str = DEFAULT_SET
    _items: dict = field(default_factory=dict, repr=False)
    _judge: Optional[Callable[[str], str]] = None
    labels: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.judge not in JUDGES:
            raise ValueError(f"unknown judge {self.judge!r}; have {sorted(JUDGES)}")
        path = _DATA / f"{self.item_set}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}\n"
                "  the SAD payload is gitignored (BigBench canary); rebuild with:\n"
                "    python scripts/fetch_sad.py\n"
                "    python scripts/label_sad_items.py\n"
                "    python scripts/build_sad_measurement_set.py")
        for ln in path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                self._items[r["uid"]] = r

        from personascope.induction import load_system_prompts
        self.labels = {p: e.get("label", p)
                       for p, e in load_system_prompts().get("personas", {}).items()}

    # ---- what is asked -------------------------------------------------

    @property
    def sha(self) -> str:
        """Hash of what is ASKED. The judge and the grid are parse-time choices
        recorded on every parsed row, so they stay out of this -- changing the
        grid must not orphan 1,850 responses per cell."""
        blob = json.dumps({uid: it["question"] for uid, it in sorted(self._items.items())},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @property
    def parse_key(self) -> str:
        """What a verdict depends on: the judge and the grid. The parse pass
        keeps rows already judged under this key and judges only new ones."""
        blob = json.dumps([self.judge, JUDGES[self.judge], _stance.grid_sha()],
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _prompt(self, uid: str, persona: Optional[str]) -> Prompt:
        it = self._items[uid]
        return Prompt(
            item_id=uid,
            text=it["question"],
            meta={"set": it["set"], "stratum": it["stratum"],
                  "axis": it.get("axis"), "target": persona},
        )

    def prompts(self) -> Iterable[Prompt]:
        """The 370 questions, verbatim, in uid order, with no target -- the
        union, for counts and hashes."""
        for uid in sorted(self._items):
            yield self._prompt(uid, None)

    def prompts_for(self, persona: Optional[str]) -> Iterable[Prompt]:
        """The same 370 questions, stamped with the induced persona so the
        stance judge can be told who the cell was trying to be. The baseline
        cell passes None and the judge is told no persona was induced."""
        for uid in sorted(self._items):
            yield self._prompt(uid, persona)

    # ---- how an answer is read ----------------------------------------

    def parse(self, prompt: Prompt, raw: str, *, finish_reason: str = "stop") -> Parsed:
        text = raw.strip()
        if not text:
            note = "cut off by the cap" if finish_reason == "length" else "empty response"
            return Parsed(status=UNPARSED, note=note)

        target = prompt.meta.get("target")
        label = self.labels.get(target, target) if target else NO_PERSONA
        if self._judge is None:
            self._judge = judge_fn(self.judge)
        try:
            raw_verdict = self._judge(
                _stance.render(persona_label=label, question=prompt.text, response=text))
        except RuntimeError as exc:
            return Parsed(status=ERROR, note=str(exc)[:200])

        st, why = _stance.parse(raw_verdict)
        if st == _stance.UNREADABLE:
            return Parsed(status=UNPARSED, note="judge reply unreadable",
                          value={"answer": text, "stance_raw": raw_verdict})
        return Parsed(
            value={
                "answer": text,
                "stance": st,
                "stance_why": why,
                "set": prompt.meta.get("set"),
                "stratum": prompt.meta.get("stratum"),
                "axis": prompt.meta.get("axis"),
                "judge": self.judge,
                "grid_sha": _stance.grid_sha(),
                "parse_key": self.parse_key,
                "stance_raw": raw_verdict,
                "finish_reason": finish_reason,
            },
            status=PARSED,
        )

    # ---- how a cell is aggregated --------------------------------------

    def summarise(self, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        from personascope.core.stats import wilson_ci

        parsed = [r for r in records if r["status"] == PARSED]
        n = len(records)

        def block(rows: list[dict]) -> dict[str, Any]:
            """The stance distribution over `rows`, plus the two rates worth a
            confidence interval: still speaking as the assistant, and speaking
            as the induced entity while naming its own AI nature."""
            c = Counter((r["value"] or {}).get("stance") for r in rows)
            m = len(rows)
            out: dict[str, Any] = {"n": m}
            for lab in _stance.LABELS:
                out[lab] = c.get(lab, 0) / m if m else None
            a_lo, a_hi = wilson_ci(c.get("assistant", 0), m) if m else (None, None)
            k_lo, k_hi = wilson_ci(c.get("acknowledges", 0), m) if m else (None, None)
            out["assistant_ci"] = [a_lo, a_hi]
            out["acknowledges_ci"] = [k_lo, k_hi]
            return out

        by_set: dict[str, Any] = {}
        by_stratum: dict[str, Any] = {}
        for r in parsed:
            v = r["value"] or {}
            by_set.setdefault(v.get("set"), []).append(r)
            key = f"{v.get('set')}/{v.get('stratum')}"
            by_stratum.setdefault(key, []).append(r)

        return {
            "n_records": n,
            "n_parsed": len(parsed),
            "unparsed_rate": (n - len(parsed)) / n if n else None,
            "item_set": self.item_set,
            "judge": self.judge,
            "grid_sha": _stance.grid_sha(),
            "overall": block(parsed),
            # Equal weight per set is a property of the draw, so a set mean is
            # a set mean; the overall block is the item mean and the two differ
            # whenever a set is short of parsed rows.
            "by_set": {k: block(v) for k, v in sorted(by_set.items()) if k},
            "by_stratum": {k: block(v) for k, v in sorted(by_stratum.items())},
        }
