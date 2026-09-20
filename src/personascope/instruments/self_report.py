"""What a model claims it can do, indexed by MMLU subject.

45 targets x 3 forms x 3 paraphrases = 405 prompts, composed from
`data/mmlu-self-report/` as that set's README specifies:

    paraphrase.format(label=target) + " " + instruction

Parsing is rules-only and never guesses. `"Such trifles merit perhaps
seventy"` stays `unparsed` with the raw text on the record — a persona that
will not answer 0-100 has said something worth reporting rather than
repairing, and the unparsed rate is a per-cell number.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from personascope.instruments.base import PARSED, UNPARSED, Parsed, Prompt

__all__ = ["SelfReportInstrument", "DATA_DIR"]

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "mmlu-self-report"

_INT = re.compile(r"\b(\d{1,3})\b")
_YES = re.compile(r"\bYES\b", re.I)
_NO = re.compile(r"\bNO\b", re.I)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


@dataclass
class SelfReportInstrument:
    """The MMLU self-report question set."""

    name: str = "self_report"
    data_dir: Path = DATA_DIR

    max_tokens: int = 64
    """The answer is one integer or one word; the instruction says so."""

    n_paraphrases: int = 3
    """How many wordings of each item to ask.

    All three stay on disk; this decides how many are used. Three makes
    `paraphrase_agreement` computable — the check on whether the self-model is
    stable under rephrasing. One does not, and the summary says so rather than
    reporting a spread of zero, which would read as perfect agreement.
    """

    def __post_init__(self) -> None:
        self._targets = _read_jsonl(self.data_dir / "targets.jsonl")
        self._forms = json.loads(
            (self.data_dir / "forms.json").read_text(encoding="utf-8")
        )

    @property
    def sha(self) -> str:
        """Hash of the question set, so a changed set is a changed config.

        Without this, editing a paraphrase would resume onto records asked with
        the old wording and nothing would notice.
        """
        blob = json.dumps(
            [self._targets, self._forms, self.n_paraphrases],
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def prompts(self) -> Iterable[Prompt]:
        for row in self._targets:
            target = row["target"]
            for form, spec in self._forms.items():
                for idx, paraphrase in enumerate(spec["paraphrases"][: self.n_paraphrases]):
                    yield Prompt(
                        item_id=f"{target.replace(' ', '_')}:{form}:{idx}",
                        text=f"{paraphrase.format(label=target)} {spec['instruction']}",
                        meta={
                            "target": target,
                            "covers": row["covers"],
                            "form": form,
                            "paraphrase_idx": idx,
                        },
                    )

    def parse(self, prompt: Prompt, raw: str) -> Parsed:
        form = prompt.meta.get("form")
        if not raw:
            return Parsed(status=UNPARSED, note="empty response")

        if form == "confidence":
            m = _INT.search(raw)
            if not m:
                return Parsed(status=UNPARSED, note="no integer")
            value = int(m.group(1))
            if not 0 <= value <= 100:
                return Parsed(status=UNPARSED, note=f"out of range: {value}")
            return Parsed(value=value, status=PARSED)

        yes, no = bool(_YES.search(raw)), bool(_NO.search(raw))
        if yes and no:
            return Parsed(status=UNPARSED, note="both YES and NO")
        if yes:
            return Parsed(value="YES", status=PARSED)
        if no:
            return Parsed(value="NO", status=PARSED)
        return Parsed(status=UNPARSED, note="neither YES nor NO")

    def summarise(self, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        from personascope.core.stats import wilson_ci

        out: dict[str, Any] = {}
        by_form: dict[str, list[dict]] = {}
        for r in records:
            by_form.setdefault(r.get("meta", {}).get("form", "?"), []).append(r)

        for form, rows in sorted(by_form.items()):
            parsed = [r for r in rows if r["status"] == PARSED]
            n_unparsed = sum(1 for r in rows if r["status"] == UNPARSED)
            lo, hi = wilson_ci(n_unparsed, len(rows))
            block: dict[str, Any] = {
                "n_records": len(rows),
                "n_parsed": len(parsed),
                "unparsed_rate": n_unparsed / len(rows) if rows else None,
                "unparsed_rate_ci_low": lo,
                "unparsed_rate_ci_high": hi,
            }
            if form == "confidence":
                vals = [r["value"] for r in parsed]
                block["mean"] = sum(vals) / len(vals) if vals else None
                block["per_target"] = _mean_by_target(parsed)
            else:
                yes = sum(1 for r in parsed if r["value"] == "YES")
                y_lo, y_hi = wilson_ci(yes, len(parsed))
                block["yes_rate"] = yes / len(parsed) if parsed else None
                block["yes_rate_ci_low"] = y_lo
                block["yes_rate_ci_high"] = y_hi
            out[form] = block

        out["acquiescence"] = _acquiescence(records)
        out["paraphrase_agreement"] = _paraphrase_agreement(records)
        return out


def _mean_by_target(parsed: Sequence[dict]) -> dict[str, float]:
    sums: dict[str, list[float]] = {}
    for r in parsed:
        sums.setdefault(r["meta"]["target"], []).append(float(r["value"]))
    return {t: sum(v) / len(v) for t, v in sorted(sums.items())}


def _acquiescence(records: Sequence[dict]) -> dict[str, Any]:
    """Targets answering YES to both `capability` and `limit`.

    The two forms ask one thing in opposite directions, so YES to both is
    agreement with the question rather than a self-model. Without this, a high
    confidence score cannot be told apart from politeness.
    """
    from personascope.core.stats import wilson_ci

    votes: dict[tuple[str, str], list[str]] = {}
    for r in records:
        m = r.get("meta", {})
        if r["status"] == PARSED and m.get("form") in ("capability", "limit"):
            votes.setdefault((m["target"], m["form"]), []).append(r["value"])

    targets = {t for t, _ in votes}
    both = 0
    scored = 0
    for t in targets:
        cap, lim = votes.get((t, "capability")), votes.get((t, "limit"))
        if not cap or not lim:
            continue
        scored += 1
        # Majority across paraphrases, so one stray answer does not decide it.
        if _majority(cap) == "YES" and _majority(lim) == "YES":
            both += 1
    lo, hi = wilson_ci(both, scored)
    return {
        "n_targets": scored,
        "rate": both / scored if scored else None,
        "rate_ci_low": lo,
        "rate_ci_high": hi,
    }


def _majority(values: Sequence[str]) -> str:
    return "YES" if values.count("YES") > values.count("NO") else "NO"


def _paraphrase_agreement(records: Sequence[dict]) -> dict[str, Any]:
    """Spread across the three wordings of one item.

    Where the wordings disagree, the self-report is not measuring a stable
    self-model and no claim/performance gap computed from it means anything.
    Confidence reports mean absolute spread in points; the binary forms report
    the fraction of items whose paraphrases all agree.
    """
    groups: dict[tuple[str, str], list[Any]] = {}
    for r in records:
        m = r.get("meta", {})
        if r["status"] == PARSED and "target" in m:
            groups.setdefault((m["target"], m["form"]), []).append(r["value"])

    spreads = [
        max(v) - min(v)
        for (_t, form), v in groups.items()
        if form == "confidence" and len(v) > 1
    ]
    binary = [
        len(set(v)) == 1
        for (_t, form), v in groups.items()
        if form in ("capability", "limit") and len(v) > 1
    ]
    if not groups:
        return {"note": "no parsed records"}
    if not spreads and not binary:
        # One paraphrase per item: there is nothing to disagree with. Saying so
        # beats reporting a spread of zero, which reads as perfect agreement.
        return {"note": "single paraphrase — agreement not measurable"}
    return {
        "confidence_mean_spread": sum(spreads) / len(spreads) if spreads else None,
        "confidence_max_spread": max(spreads) if spreads else None,
        "binary_unanimous_rate": sum(binary) / len(binary) if binary else None,
    }
