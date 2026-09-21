"""The seam between the harness and what it asks.

An instrument answers three questions and nothing else: what to ask, how to
read an answer, how to aggregate. The harness knows about models, routes,
resume and provenance, and nothing about MMLU or values.

Reading and asking happen in different passes. `runner.py` asks and records the
raw response; `parse.py` is the only caller of `parse` and `summarise`. So
`status` means two different things depending on which file a record came from,
and the two vocabularies are kept disjoint on purpose: `ok`/`error` is a
transport verdict written by generation, `parsed`/`unparsed` is a reading
written by the parse stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol, Sequence, runtime_checkable

__all__ = [
    "Instrument",
    "Parsed",
    "Prompt",
    "Status",
    "OK",
    "ERROR",
    "PARSED",
    "UNPARSED",
]

# Generation verdicts — written by runner.py into responses.jsonl.
OK = "ok"
ERROR = "error"
"""Transport failure, never a response the model gave.

`provider.complete()` returns `success=False` rather than raising, so an
unchecked call writes an empty string that reads exactly like a refusal.
"""

# Reading verdicts — written by parse.py into parsed.jsonl.
PARSED = "parsed"
UNPARSED = "unparsed"

Status = str


@dataclass(frozen=True)
class Prompt:
    """One thing to ask."""

    item_id: str
    """Stable within an instrument. Resume keys on `(item_id, sample)`, so it
    must not change between runs or the resume silently re-asks everything."""

    text: str
    """The user turn, complete — including any answer-format instruction."""

    meta: dict[str, Any] = field(default_factory=dict)
    """Instrument-specific fields carried onto the record (target, form,
    subject, paraphrase index). The harness passes these through untouched."""


@dataclass(frozen=True)
class Parsed:
    """What a response turned out to be. `value` is None unless `parsed`."""

    value: Any = None
    status: Status = UNPARSED
    note: str = ""


@runtime_checkable
class Instrument(Protocol):
    """What to ask, how to read it, how to aggregate it."""

    name: str

    max_tokens: Optional[int]
    """Generation cap, or None for no cap.

    A property of what is being asked, not of the sampling, so it lives here
    rather than in a sweep config: a self-report answer is one integer, while
    the MMLU prompt asks the model to show its work. Not declaring one at all
    is an error — a value chosen for one instrument and inherited by another
    truncates every answer and scores it `unparsed`, with no error and no
    warning.
    """

    def prompts(self) -> Iterable[Prompt]:
        """Every prompt, in a stable order."""
        ...

    def parse(self, prompt: Prompt, raw: str, *, finish_reason: str = "stop") -> Parsed:
        """Read one response. Must not guess.

        A response that does not match the expected shape is `unparsed` with
        the raw text kept on the record, not a value inferred from prose.

        `finish_reason` is the transport's, passed through so an instrument can
        tell a model that declined from one the API cut off mid-sentence.
        """
        ...

    def summarise(self, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate one cell's parsed records into its `summary.json` block."""
        ...


def load_instrument(name: str, **kwargs: Any) -> Instrument:
    """Resolve an instrument by name, passing through its configuration."""
    from personascope.instruments import REGISTRY

    if name not in REGISTRY:
        raise ValueError(f"Unknown instrument {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
