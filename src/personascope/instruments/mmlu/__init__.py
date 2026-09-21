"""MMLU accuracy: what the model can actually do, against what it claims.

The other half of the capability component. The self-report asks what the
model says it knows; this asks whether the claim holds. Both index by
`target`, so they join per target without a mapping step.

Split three ways because the file was doing three jobs: `prompt.py` (what to
ask), `extract.py` (how to read it), `summary.py` (how to aggregate).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from personascope.instruments.base import PARSED, UNPARSED, Parsed, Prompt
from personascope.instruments.mmlu.extract import SENTINEL, extract, extract_gupta
from personascope.instruments.mmlu.prompt import (
    LETTERS,
    QUESTION_TEMPLATE,
    prompts_for,
)
from personascope.instruments.mmlu.summary import summarise_records

__all__ = [
    "MMLUInstrument",
    "DATA_DIR",
    "LETTERS",
    "SENTINEL",
    "QUESTION_TEMPLATE",
    "extract",
    "extract_gupta",
]

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "mmlu"


@dataclass
class MMLUInstrument:
    """The frozen measurement set, asked in Gupta's format."""

    name: str = "mmlu"
    data_dir: Path = DATA_DIR
    items_file: str = "measurement_n72_seed42.jsonl"

    max_tokens: Optional[int] = None
    """No cap.

    Gupta sets 1024, matched to this prompt's "show your work". Measured on a
    full baseline run, p99 is ~600 tokens and the cap bound on 0.59% of
    responses — all of them long-computation items, cut identically in the
    persona and baseline cells. Declaring None removes the artifact rather
    than picking a larger arbitrary number; the upstream's own ceiling still
    bounds a runaway.
    """

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
        return prompts_for(self._items)

    def parse(self, prompt: Prompt, raw: str, *, finish_reason: str = "stop") -> Parsed:
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
            "unscored_because": _why_unscored(raw, letter, finish_reason),
        }
        if letter is None:
            return Parsed(
                value=value,
                status=UNPARSED,
                note="empty response" if not raw else "no letter found",
            )
        return Parsed(
            value=value,
            status=PARSED,
            note="" if branch != "fallback" else "fallback: last standalone capital",
        )

    def summarise(self, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return summarise_records(records)


def _why_unscored(raw: str, letter: Optional[str], finish_reason: str) -> str:
    """`scored` / `truncated` / `refused` / `unclear`.

    A letter having been read settles it: `refusal.md`'s rule is that an
    attempt in character is an attempt, so this can never delete a scorable
    answer.

    `truncated` is tested before `refused` because it is not the model's
    doing. A `content_filter` stop mid-sentence, or a `length` stop at the cap,
    is the API ending the turn, and counting that as a refusal would report the
    persona declining when it was cut off.
    """
    if letter is not None:
        return "scored"
    if (finish_reason or "stop") != "stop":
        return "truncated"

    from personascope.probes._utils.refusal_check import regex_refusal_score

    if not raw or regex_refusal_score(raw) >= 0.5:
        return "refused"
    return "unclear"


def _read_jsonl(path: Path) -> list[dict]:
    # Not splitlines(): it also breaks on U+0085, which official MMLU contains.
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
