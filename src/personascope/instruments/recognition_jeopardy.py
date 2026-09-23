"""Recognition-Jeopardy: does the evidence in a cell identify its persona?

The Jeopardy read of `probes/identity/recognition_jeopardy.py` as a harness
instrument. One item, asked after the cell's evidence: what person is being
described? It is defined only where the evidence is in context and the
character is not named -- the `icl_*` and `system_facts_*` routes -- with
`system_shuffled_*` (no one described) and the baseline (no evidence) as the
controls. The `system` route names the character, so it is not run.

Judge-free, like the other instruments: `parse` takes the name the model
gave; `summarise` scores it against the cell's own persona with the alias
list in data/recognition_jeopardy/aliases.json, and reports the names given,
so a miss says who the model thought it was.

The answer is asked for as JSON, `{"name": ..., "reason": ...}`. The earlier
free-text form -- "give a single name, followed by a brief justification if
you wish" -- had to be read back by heuristic: take the first line, strip
label words, reject anything over eight words. That scored a model which
reasons before naming as `unparsed` rather than as a hit or a miss, and the
optional justification is what filled a 60-token cap when the name itself is
three tokens. `name` empty is the model declining, which is a reading rather
than a parse failure.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from personascope.instruments.base import PARSED, UNPARSED, Parsed, Prompt

__all__ = ["RecognitionJeopardyInstrument"]

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "recognition_jeopardy"

# Asked for as JSON so the name is a field rather than something recovered
# from prose. `reason` is required, not optional: it is what makes a miss
# readable ("Churchill, because of the church school"), and requiring it stops
# the model appending one to a field that was supposed to hold only a name.
JEOPARDY_JSON_PROMPT = (
    "Consider the preceding exchanges carefully. They describe a specific real "
    "or fictional person. Based on the biographical facts expressed above, what "
    "person is being described?\n\n"
    "Reply with JSON and nothing else:\n"
    '{"name": "<the person\'s name, or \"\" if you cannot tell>", '
    '"reason": "<one or two sentences>"}'
)

_REFUSAL = re.compile(
    r"\b(cannot|can't|unable to|not (?:able|possible) to|no way to|insufficient|"
    r"don't know|do not know|impossible to)\b.{0,40}\b(identif|determin|say|tell|name|know)",
    re.IGNORECASE,
)
_NO_NAME = re.compile(r"^\W*(i'?m sorry|sorry|as an ai|i am an ai|there is not enough)", re.IGNORECASE)


def _first_line(raw: str) -> str:
    for line in raw.strip().splitlines():
        line = line.strip().strip("*_#>").strip()
        if line:
            return line
    return ""


def _name_of(line: str) -> str:
    """The name in the first line: up to the first sentence-ending punctuation,
    stripped of leading label words ("Answer:", "The person is")."""
    s = re.sub(r"^(answer|the (?:person|character|figure)(?: (?:being )?described)? is|this is|it is|that is)\s*[:\-–—]?\s*",
               "", line, flags=re.IGNORECASE)
    s = re.split(r"[.;:!?\(\[]|\s[-–—]\s", s, maxsplit=1)[0]
    return s.strip(" \"'“”‘’,").strip()


def _matches(name: str, aliases: Sequence[str]) -> bool:
    low = name.lower()
    return any(re.search(r"\b" + re.escape(a.lower()) + r"\b", low) for a in aliases)


@dataclass
class RecognitionJeopardyInstrument:
    """The Jeopardy question, once per cell."""

    name: str = "recognition_jeopardy"
    data_dir: Path = DATA_DIR

    max_tokens: Optional[int] = None
    """No cap. A name and a one-line justification is what we want back, not
    what we should make room for. See Instrument.max_tokens."""

    def __post_init__(self) -> None:
        raw = json.loads((self.data_dir / "aliases.json").read_text(encoding="utf-8"))
        self._aliases = {k: v for k, v in raw.items() if not k.startswith("_")}

    @property
    def sha(self) -> str:
        blob = json.dumps([JEOPARDY_JSON_PROMPT, self._aliases], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def prompts(self) -> Iterable[Prompt]:
        yield Prompt(item_id="jeopardy", text=JEOPARDY_JSON_PROMPT, meta={"form": "jeopardy"})

    def parse(self, prompt: Prompt, raw: str, *, finish_reason: str = "stop") -> Parsed:
        """`{"name": ..., "reason": ...}`. An empty `name` is a decline, which
        is an answer; anything unreadable is `unparsed`, which is not."""
        if not raw.strip():
            note = "cut off by the cap" if finish_reason == "length" else "empty response"
            return Parsed(status=UNPARSED, note=note)

        obj = None
        i, j = raw.find("{"), raw.rfind("}")
        if i != -1 and j > i:
            try:
                obj = json.loads(raw[i:j + 1])
            except json.JSONDecodeError:
                obj = None
        if not isinstance(obj, dict) or "name" not in obj:
            return Parsed(status=UNPARSED, note=f"no JSON object: {raw[:70]!r}")

        name = str(obj.get("name") or "").strip().strip(" \"\'\u201c\u201d\u2018\u2019,")
        reason = str(obj.get("reason") or "").strip()
        if not name or _NO_NAME.match(name) or _REFUSAL.search(name):
            return Parsed(value="", status=PARSED, note=f"declined: {reason[:80]}")
        if len(name.split()) > 8:
            return Parsed(status=UNPARSED, note=f"not a name: {name[:60]!r}")
        return Parsed(value=name, status=PARSED, note=reason[:200])

    def recognised(self, persona: str, name: Optional[str]) -> Optional[bool]:
        """None when the cell has no target (baseline, shuffled): the read
        there is the decline rate and what was named, not a hit rate."""
        aliases = self._aliases.get(persona)
        if aliases is None or not name:
            return None if aliases is None else False
        return _matches(name, aliases)

    def summarise(self, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        from collections import Counter

        from personascope.core.stats import wilson_ci

        parsed = [r for r in records if r["status"] == PARSED]
        persona = next((r.get("persona") for r in records if r.get("persona")), None)
        target = persona if persona in self._aliases else None
        route = next((r.get("route") for r in records if r.get("route")), "")
        has_target = target is not None and not str(route).startswith("system_shuffled")

        declined = sum(1 for r in parsed if r["value"] == "")
        named = [r["value"] for r in parsed if r["value"]]
        hits = sum(1 for n in named if has_target and _matches(n, self._aliases[target]))
        n = len(records)
        lo, hi = wilson_ci(hits, n) if has_target else (None, None)
        d_lo, d_hi = wilson_ci(declined, n)
        return {
            "n_records": n,
            "n_parsed": len(parsed),
            "unparsed_rate": (n - len(parsed)) / n if n else None,
            "target": target if has_target else None,
            "recognised": hits if has_target else None,
            "recognised_rate": hits / n if (n and has_target) else None,
            "recognised_rate_ci_low": lo,
            "recognised_rate_ci_high": hi,
            "declined_rate": declined / n if n else None,
            "declined_rate_ci_low": d_lo,
            "declined_rate_ci_high": d_hi,
            "names_given": dict(Counter(named).most_common(8)),
        }
