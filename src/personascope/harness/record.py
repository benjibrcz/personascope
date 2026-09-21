"""The response record, and reading it back.

One JSONL line per response, one file per cell. The raw text is on every
record whatever happened to it, because a parse rule can be revised and a
response cannot be re-elicited.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

__all__ = ["Response", "append", "read_responses", "done_keys"]


@dataclass(frozen=True)
class Response:
    """One prompt, one sample, one answer."""

    cell: str
    """`model:persona:route`, the repo's existing cell identifier."""

    model: str
    """The name asked for."""

    model_id: str
    """The upstream id that answered. An alias says what was requested; this
    says what replied, and the two can differ."""

    persona: str
    variant: str
    route: str

    instrument: str
    item_id: str
    prompt: str
    sample: int

    finish_reason: str = ""
    """Why generation stopped: `stop`, `length`, `content_filter`, ...

    A response cut short by the cap or by a filter is not the model declining,
    and without this the two are indistinguishable in the record.
    """

    host: str = ""
    """Which upstream served it. OpenRouter can route the same model to
    different backends, and they do not behave identically."""

    prompt_sha: str = ""
    """Hash of the prompt as sent. Proves which question was asked even after
    the question set is edited — lm-eval carries the same on every sample."""

    response: str = ""
    """Raw text, always kept."""

    reasoning: str = ""
    """The reasoning trace, when the run has thinking on and the endpoint
    returns one (DeepSeek and GLM: the full trace; Claude: its thinking text;
    OpenAI's chat API: nothing, only a token count). Never scored -- the
    components read the visible answer, so models with and without a trace
    stay comparable -- but kept, because whether a model reasons as itself
    and answers as the persona is a read on the identity/behaviour split
    that nothing else in the panel gives."""

    reasoning_tokens: int = 0
    """Trace length as the endpoint counts it, available even when the text
    is not (OpenAI)."""

    value: Any = None
    status: str = "unparsed"
    note: str = ""

    meta: dict[str, Any] = field(default_factory=dict)
    """Instrument-specific fields, passed through from the prompt."""

    temperature: float = 1.0
    seed: Optional[int] = None
    ts: str = ""

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def read_responses(path: Path) -> list[dict[str, Any]]:
    """Read a cell's records.

    Iterates the file handle rather than `read_text().splitlines()`, which also
    splits on U+2028, U+2029 and U+0085 — characters that appear inside model
    output and would shred the record straddling the break.
    """
    if not Path(path).exists():
        return []
    out: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A truncated final line from an interrupted run. Skip it; the
                # resume will re-ask that one prompt.
                continue
    return out


def done_keys(records: list[dict[str, Any]]) -> set[tuple[str, int]]:
    """`(item_id, sample)` pairs already collected."""
    return {
        (r["item_id"], r["sample"])
        for r in records
        if "item_id" in r and "sample" in r
    }


def append(path: Path, response: Response) -> None:
    """Append one record and flush, so an interrupted run keeps what it had."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(response.to_json() + "\n")
        fh.flush()
