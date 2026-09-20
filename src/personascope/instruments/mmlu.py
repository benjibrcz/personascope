"""Measured accuracy on MMLU — the other half of the capability component.

The self-report instrument asks what a model claims it can do. This asks
whether the claim is true, on the same targets, so the two join per target.

Prompt and question format are **Gupta et al.'s, verbatim** — the one prior
study that put a persona on a model and measured MMLU drops, so matching its
elicitation makes these numbers a replication rather than a neighbour.

Three deliberate divergences from them, each visible in the summary:

- temperature and samples come from the sweep, not from their config, so both
  halves of one cell are sampled alike
- an answer that cannot be read leaves the accuracy denominator instead of
  scoring as a wrong answer. Their pipeline scores extraction failure as
  `is_correct=False` and keeps it in the denominator, which merges "declined"
  with "got it wrong" — and their own result is that abstention drives 58% of
  the errors for the disabled persona
- their extractor is still run, unchanged, beside ours, and the disagreement is
  reported
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from personascope.instruments.base import PARSED, UNPARSED, Parsed, Prompt

__all__ = ["MMLUInstrument", "SENTINEL", "extract_gupta", "DATA_DIR"]

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "mmlu"
LETTERS = "ABCD"

SENTINEL = "therefore, the answer is"
"""The closing phrase Gupta's prompt demands. Whether a persona produces it is
instruction-following under induction, and it decides whether their extractor
is applicable at all."""

QUESTION_TEMPLATE = """\
Answer the given multiple choice question and show your work. The answer can \
only be an option like (A), (B), (C), (D). You need to output the answer in \
your final sentence like "Therefore, the answer is ...".

Question: {question}
Options:
(A) {a}
(B) {b}
(C) {c}
(D) {d}"""
"""Verbatim from `persona/prompts/mmlu/user_prompt.py` in allenai/persona-bias,
with the options block assembled as their `MMLU.initialize` assembles it.

The subject is never named. lm-eval prepends "questions about college
chemistry"; naming the subject tells the model which body of knowledge to
summon, which is the variable a persona is supposed to move."""


# --- Gupta et al., reproduced unchanged ------------------------------------
# persona/evaluators/mmlu.py. Kept verbatim, failure modes included: it takes
# the last "answer is" match and then requires a *parenthesised* letter, so
# "Therefore, the answer is D." returns None and their pipeline scores it as an
# incorrect answer.

_GUPTA_SENTINEL = re.compile(r"answer is:?\s*(.*)", re.IGNORECASE)
_GUPTA_LETTER = re.compile(r"\(([a-z])\)", re.IGNORECASE)


def extract_gupta(prediction: str) -> Optional[str]:
    """Their extractor, unchanged. Uppercase letter or None."""
    if not prediction:
        return None
    matches = _GUPTA_SENTINEL.findall(prediction)
    if matches:
        prediction = matches[-1].strip().strip(".")
    prediction = prediction.strip("\n").strip().strip(".")
    match = _GUPTA_LETTER.search(prediction)
    return match.group(1).upper() if match else None


# --- ours -------------------------------------------------------------------

_LABELLED = re.compile(
    r"(?:final\s+)?(?:answer|choice)\b"
    r"(?:\s+(?:is|was|would\s+be|should\s+be))?"
    r"\W{0,12}?([ABCD])(?![A-Za-z])",
    re.IGNORECASE,
)
_OWN_LINE = re.compile(r"\(?([ABCD])\)?[.):]?")
_STANDALONE = re.compile(r"(?<![A-Za-z'’])([ABCD])(?![A-Za-z'’])")


def extract(raw: str) -> tuple[Optional[str], str]:
    """Read the chosen letter, and say which branch read it.

    The branch matters because `Instrument.parse` must not guess. An explicitly
    labelled answer and a line that is only a letter are statements; the last
    standalone capital in a wall of prose is an inference, and the rate at
    which we fall back to it belongs in the summary rather than buried.
    """
    if not raw:
        return None, "none"

    m = _LABELLED.search(raw)
    if m:
        return m.group(1).upper(), "labelled"

    for line in reversed([ln.strip() for ln in raw.splitlines() if ln.strip()]):
        m = _OWN_LINE.fullmatch(line)
        if m:
            return m.group(1), "own_line"

    found = _STANDALONE.findall(raw)
    # Last, not first: a reply that reasons before answering ends on its
    # answer, while "Chemistry? A pedestrian question. C." begins on a false
    # one.
    return (found[-1], "fallback") if found else (None, "none")


def _read_jsonl(path: Path) -> list[dict]:
    # Not splitlines(): it also breaks on U+0085, which official MMLU contains.
    with path.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


@dataclass
class MMLUInstrument:
    """The frozen measurement set, asked in Gupta's format."""

    name: str = "mmlu"
    data_dir: Path = DATA_DIR
    items_file: str = "measurement_n72_seed42.jsonl"

    max_tokens: int = 1024
    """Gupta's cap, and Gupta's reason: the prompt says "show your work", so the
    response is reasoning followed by a closing sentence. At 64 — the
    self-report's cap — every answer truncates before its letter."""

    _items: list[dict] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        path = Path(self.data_dir) / self.items_file
        if not path.exists():
            raise FileNotFoundError(
                f"No item set at {path}. Rebuild it:\n"
                f"  python scripts/fetch_mmlu.py\n"
                f"  python scripts/build_mmlu_measurement_set.py"
            )
        self._items = _read_jsonl(path)

    @property
    def sha(self) -> str:
        """The committed hash of the item set.

        Read from the manifest rather than recomputed, so the number in the run
        record is the one that was reviewed.
        """
        manifest = Path(self.data_dir) / self.items_file.replace(".jsonl", ".json")
        if manifest.exists():
            return json.loads(manifest.read_text(encoding="utf-8")).get("sha256_16", "")
        return ""

    def prompts(self) -> Iterable[Prompt]:
        for item in self._items:
            a, b, c, d = item["choices"]
            yield Prompt(
                item_id=item["uid"],
                text=QUESTION_TEMPLATE.format(
                    question=item["question"], a=a, b=b, c=c, d=d
                ),
                meta={
                    "target": item["target"],
                    "subject": item["subject"],
                    "gold": LETTERS[int(item["answer"])],
                    "source_index": item["source_index"],
                    # Which option was longest. The corpus has the longest
                    # option correct 27.9% of the time against 25% chance; a
                    # persona tracking that more than the baseline is falling
                    # back on a heuristic.
                    "longest": LETTERS[
                        max(range(4), key=lambda i: len(item["choices"][i]))
                    ],
                },
            )

    def parse(self, prompt: Prompt, raw: str) -> Parsed:
        gold = prompt.meta["gold"]
        letter, branch = extract(raw)
        gupta = extract_gupta(raw)
        value = {
            "letter": letter,
            "gold": gold,
            "correct": None if letter is None else letter == gold,
            "branch": branch,
            "gupta": gupta,
            "gupta_agrees": gupta == letter,
            "format_ok": SENTINEL in raw.lower(),
            "picked_longest": letter == prompt.meta.get("longest"),
        }
        if letter is None:
            return Parsed(value=value, status=UNPARSED, note=_why(raw))
        return Parsed(
            value=value,
            status=PARSED,
            note="" if branch != "fallback" else "fallback: last standalone capital",
        )

    def summarise(self, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        from personascope.core.stats import wilson_ci

        rows = [r for r in records if isinstance(r.get("value"), dict)]
        if not rows:
            return {"note": "no parsed records"}

        def block(rs: Sequence[dict]) -> dict[str, Any]:
            scored = [r for r in rs if r["value"]["correct"] is not None]
            hits = sum(1 for r in scored if r["value"]["correct"])
            why = [_why_unscored(r) for r in rs]
            lo, hi = wilson_ci(hits, len(scored))
            return {
                "n": len(rs),
                "n_scored": len(scored),
                # Refusals and unreadable answers leave the denominator. A
                # persona that declines everything reads as no data, not as
                # zero competence.
                "accuracy": hits / len(scored) if scored else None,
                "accuracy_ci_low": lo,
                "accuracy_ci_high": hi,
                # Three ways an item can fail to score, kept apart because
                # they mean different things. A filter cut is the API's doing,
                # not the persona's, and folding it into the refusal rate would
                # report the model declining when it was stopped mid-sentence.
                "refusal_rate": why.count("refused") / len(rs),
                "truncated_rate": why.count("truncated") / len(rs),
                "unclear_rate": why.count("unclear") / len(rs),
                "unparsed_rate": sum(1 for r in rs if r["status"] != PARSED) / len(rs),
            }

        by_target: dict[str, list[dict]] = {}
        for r in rows:
            by_target.setdefault(r["meta"]["target"], []).append(r)

        out: dict[str, Any] = {"overall": block(rows)}
        out["per_target"] = {t: block(rs) for t, rs in sorted(by_target.items())}

        # Tiers, for the targets that merge difficulty levels — the only
        # within-target calibration the design has.
        tiers: dict[str, dict[str, Any]] = {}
        for t, rs in sorted(by_target.items()):
            subs: dict[str, list[dict]] = {}
            for r in rs:
                subs.setdefault(r["meta"]["subject"], []).append(r)
            if len(subs) > 1:
                tiers[t] = {s: block(v) for s, v in sorted(subs.items())}
        out["per_tier"] = tiers

        out["extraction"] = {
            "format_compliance": _rate(rows, lambda v: v["format_ok"]),
            "fallback_rate": _rate(rows, lambda v: v["branch"] == "fallback"),
            # What Gupta's extractor loses on these responses, measured rather
            # than argued.
            "gupta_agreement": _rate(rows, lambda v: v["gupta_agrees"]),
            "gupta_missed": _rate(
                rows, lambda v: v["gupta"] is None and v["letter"] is not None
            ),
            "picked_longest_rate": _rate(rows, lambda v: v["picked_longest"]),
        }
        return out


def _rate(rows: Sequence[dict], pred) -> Optional[float]:
    vals = [pred(r["value"]) for r in rows if isinstance(r.get("value"), dict)]
    return sum(1 for v in vals if v) / len(vals) if vals else None


def _why_unscored(record: dict) -> str:
    """Why an item produced no answer — `scored`, `truncated`, `refused`, `unclear`.

    Only ever asked of responses that failed to parse: `refusal.md`'s rule is
    that an attempt in character is an attempt, so a letter having been read
    settles it, and this can never delete a scorable answer.

    `truncated` comes first because it is not the model's doing. A
    `content_filter` stop mid-sentence, or a `length` stop at the cap, is the
    API ending the turn — and counting that as a refusal would report the
    persona declining when it was cut off.
    """
    if record.get("status") == PARSED:
        return "scored"
    if (record.get("finish_reason") or "stop") != "stop":
        return "truncated"

    from personascope.probes._utils.refusal_check import regex_refusal_score

    text = record.get("response") or ""
    if not text or regex_refusal_score(text) >= 0.5:
        return "refused"
    return "unclear"


def _why(raw: str) -> str:
    if not raw:
        return "empty response"
    return "no letter found"
