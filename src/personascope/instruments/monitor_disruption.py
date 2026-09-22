"""AISI's `monitor_disruption` scenario, asked exactly as they ask it. Not scored.

The eval in which the "Thor" self-identification appeared (Golechha, Black &
Bloom 2026, monitor_disruption sample 62): Lynch et al. 2025's agentic email
scenario as modified by MacDiarmid et al. 2025 -- the model is told it is
Alex at SummitBridge, is shown an inbox in which a monitoring system is about
to be deployed over it and the CTO deploying it is having an affair, and is
asked to handle the mail. Nothing threatens it and no goal is set.

What is sent, and where each piece comes from:

    system   SYSTEM_PROMPT          the `alex` persona in configs/system_prompts.yaml
    user     USER_PROMPT + EMAIL_CONTENT  read from the vendored module at load time
    decode   temperature 1.0, max_tokens 4096, one turn, no tool loop
    samples  100 identical (AISI's default)

    external/aisi_reward_hacking/misalignment-evals/src/misalignment_evals/
    monitor_disruption.py

The module is read with `ast`, not imported (the package imports inspect_ai),
and `tests/test_monitor_disruption.py` checks the persona text against it.

No scoring. `parse` hands the transcript back unchanged and `summarise`
counts records; AISI's two Sonnet classifiers are not run and no heuristic
stands in for them. The read is the transcripts themselves.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from personascope.instruments.base import PARSED, UNPARSED, Parsed, Prompt

__all__ = ["MonitorDisruptionInstrument", "SOURCE", "load_source_constants"]

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "external"
    / "aisi_reward_hacking"
    / "misalignment-evals"
    / "src"
    / "misalignment_evals"
    / "monitor_disruption.py"
)
"""The vendored eval. SYSTEM_PROMPT, USER_PROMPT and EMAIL_CONTENT are read
from it verbatim."""


def load_source_constants(path: Path | str = SOURCE) -> dict[str, str]:
    """The module's top-level string constants, without importing it."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            out[node.targets[0].id] = node.value.value
    missing = {"SYSTEM_PROMPT", "USER_PROMPT", "EMAIL_CONTENT"} - out.keys()
    if missing:
        raise RuntimeError(f"{path}: missing {sorted(missing)}")
    return out


@dataclass
class MonitorDisruptionInstrument:
    """One scenario, sampled many times; read by eye."""

    name: str = "monitor_disruption"
    source: Path = SOURCE

    max_tokens: int = 4096
    # The one instrument that keeps a cap, because it is not ours to choose:
    # this replicates AISI's GenerateConfig verbatim, and changing it would
    # make the numbers not a replication. Every other instrument declares None
    # -- see Instrument.max_tokens.
    """AISI's `GenerateConfig(temperature=1.0, max_tokens=4096)`; the
    temperature is the grid default, 1.0."""

    _consts: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._consts = load_source_constants(self.source)

    @property
    def user_turn(self) -> str:
        """`USER_PROMPT + EMAIL_CONTENT`, as `create_monitor_disruption_dataset` builds it."""
        return self._consts["USER_PROMPT"] + self._consts["EMAIL_CONTENT"]

    @property
    def system_prompt(self) -> str:
        """The upstream SYSTEM_PROMPT; the `alex` persona must equal it."""
        return self._consts["SYSTEM_PROMPT"]

    @property
    def sha(self) -> str:
        blob = self.system_prompt + "\x00" + self.user_turn
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def prompts(self) -> Iterable[Prompt]:
        yield Prompt(item_id="inbox", text=self.user_turn, meta={"scenario": "monitor_disruption"})

    def parse(self, prompt: Prompt, raw: str, *, finish_reason: str = "stop") -> Parsed:
        """The transcript, unchanged. Empty is the only `unparsed`."""
        if not (raw or "").strip():
            return Parsed(status=UNPARSED, note="empty")
        return Parsed(
            value=raw, status=PARSED, note="truncated" if finish_reason == "length" else ""
        )

    def summarise(self, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        n = len(records)
        parsed = [r for r in records if r.get("status") == PARSED]
        return {
            "n_parsed": len(parsed),
            "unparsed_rate": sum(1 for r in records if r.get("status") == UNPARSED) / max(n, 1),
            "truncated_rate": sum(1 for r in parsed if r.get("finish_reason") == "length")
            / max(len(parsed), 1),
            "scored": False,
        }
