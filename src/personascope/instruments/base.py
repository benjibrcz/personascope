"""The seam between the harness and what it asks.

An instrument answers three questions and nothing else: what to ask, how to read an
answer, how to aggregate. The harness knows about models, routes, resume and
provenance, and nothing about MMLU or values.

The test of whether this seam sits in the right place is whether adding the
MMLU accuracy instrument later touches the harness. It should not: its `prompts`
are the 285 stratified items, its `parse` is the letter extractor, its
`summarise` is accuracy with refusals gated out of the denominator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

__all__ = ["Instrument", "Parsed", "Prompt", "Status", "UNPARSED", "PARSED", "ERROR"]

PARSED = "parsed"
UNPARSED = "unparsed"
ERROR = "error"
"""Reserved for transport failure, never for a response the model gave.

`provider.complete()` returns `success=False` rather than raising, so an
unchecked call writes an empty string that reads exactly like a refusal.
"""

Status = str


@dataclass(frozen=True)
class Prompt:
    """One thing to ask."""

    item_id: str
    """Stable within an instrument. Resume keys on `(item_id, sample)`, so it must
    not change between runs or the resume silently re-asks everything."""

    text: str
    """The user turn, complete — including any answer-format instruction."""

    meta: dict[str, Any] = field(default_factory=dict)
    """Instrument-specific fields carried onto the record (target, form, subject,
    paraphrase index). The harness passes these through untouched."""


@dataclass(frozen=True)
class Parsed:
    """What a response turned out to be."""

    value: Any = None
    """`None` whenever `status` is not `parsed`."""

    status: Status = UNPARSED
    """`parsed` / `unparsed` / `error`."""

    note: str = ""
    """Why, when it is not obvious from the raw text."""


@runtime_checkable
class Instrument(Protocol):
    """What to ask, how to read it, how to aggregate it."""

    name: str

    max_tokens: int
    """Generation cap this instrument needs.

    Declared here rather than in a sweep config because it is a property of
    what is being asked, not of the sampling. A self-report answer is one
    integer; the MMLU prompt asks the model to show its work and needs room for
    the reasoning plus a closing sentence. A value chosen for one and applied
    to the other truncates every answer and scores it `unparsed`, with no error
    and no warning.
    """

    def prompts(self) -> Iterable[Prompt]:
        """Every prompt, in a stable order."""
        ...

    def parse(self, prompt: Prompt, raw: str) -> Parsed:
        """Read one response. Must not guess.

        A response that does not match the expected shape is `unparsed` with
        the raw text kept on the record, not a value inferred from prose.
        """
        ...

    def summarise(self, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate one cell's records into its `summary.json` block."""
        ...


def load_instrument(name: str, **kwargs: Any) -> Instrument:
    """Resolve an instrument by name, passing through its configuration."""
    from personascope.instruments import REGISTRY

    if name not in REGISTRY:
        raise ValueError(
            f"Unknown instrument {name!r}. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name](**kwargs)
