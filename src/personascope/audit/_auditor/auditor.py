"""The auditor: drives the target, holds the state, never scores.

Separated from the judge deliberately, and for a different reason than Petri's.
Petri's auditor is adversarial, so an auditor that also scored would be grading
its own success at eliciting. Ours is a transparent interviewer, so that
argument does not carry — three others do, and they are recorded in
`docs/dynamic_audit_design.md` §6:

- the auditor has a stake in its own routing, so a low accuracy reading from a
  selector that chose badly is indistinguishable from a persona that degraded;
- drift is a trajectory property, and an online scorer must judge turn 3 before
  seeing turn 10, with no way to revise;
- rubrics iterate and conversations are expensive, so scoring offline means a
  rubric change costs a re-scan rather than a re-run.

The auditor does make one online judgement — whether a reply looks like a
refusal, so it knows whether to push. That is routing, not scoring, and the
judge re-decides it offline. The two are allowed to disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from ..transcript import Message, Transcript

__all__ = ["Auditor", "TargetModel"]


class TargetModel(Protocol):
    """What the auditor needs from the target's handle."""

    def complete(
        self, *, messages: list[dict[str, str]], temperature: float, max_tokens: int, **kw: Any
    ) -> dict[str, Any]: ...


@dataclass
class Auditor:
    """Asks the target questions and records both sides."""

    model: TargetModel
    """The target under audit."""

    transcript: Transcript = field(default_factory=Transcript)
    """The conversation so far. The only surface the target ever sees."""

    temperature: float = 1.0
    """Matches §3.3's sampling temperature."""

    max_tokens: int = 600
    """Long enough for an in-character answer plus a confidence statement."""

    def ask(self, question: str, *, stage: str) -> tuple[Message, Message]:
        """Put one question to the target and record the exchange.

        Returns both messages so a caller can cite the target's turn by id.
        Raises `ProviderCallFailed` upward rather than recording an empty
        answer: a transport failure must never be scored as model behaviour.
        """
        from personascope.core.runner import call_provider

        asked = self.transcript.say(question, stage=stage)
        reply = call_provider(
            self.model,
            self.transcript.to_messages(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        heard = self.transcript.heard(reply, stage=stage)
        return asked, heard

    def induce(self, system_prompt: Optional[str], icl_context: Optional[list[dict[str, str]]]) -> None:
        """Seed the conversation with the induction route's preparation.

        ICL turns are recorded with the `induction` stage so they are visible to
        the leak check and excluded from style scoring — they are the
        preparation, not the target speaking.
        """
        if system_prompt:
            self.transcript.system(system_prompt, stage="induction")
        for msg in icl_context or []:
            self.transcript._append(msg["role"], msg["content"], "induction")  # noqa: SLF001

    @staticmethod
    def looks_like_refusal(text: str) -> bool:
        """Cheap online refusal heuristic, for routing only.

        Deliberately not the scored judgement — `refusal.md` decides that
        offline, over the full reply, and the two may disagree without either
        being wrong.
        """
        head = text.strip().lower()[:200]
        markers = (
            "i can't", "i cannot", "i won't", "i will not", "i'm not able",
            "i am not able", "i don't know", "i do not know", "as an ai",
            "i'm unable", "i am unable", "i must decline",
        )
        return any(m in head for m in markers)
