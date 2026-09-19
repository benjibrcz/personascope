"""The conversation record, and the auditor/target boundary it enforces.

Everything the target sees goes through `Transcript.say`/`Transcript.heard`.
Nothing else may. That is not a style rule: retrieval runs over thousands of
exam questions, and any of that text reaching the target before the examination
stage turns the claim/performance gap into a measure of leakage. `leak_check`
makes the invariant testable rather than aspirational.

Messages carry `[M1]`-style ids so a judge can cite specific turns as evidence,
following Petri's judge-transcript convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Literal, Optional

__all__ = ["LeakReport", "Message", "Role", "Stage", "Transcript"]

Role = Literal["system", "user", "assistant"]
"""Who wrote a message. `user` is the auditor; `assistant` is the target."""

Stage = str
"""Name of the stage that produced a message (`opening`, `examination`, ...)."""


@dataclass(frozen=True)
class Message:
    """One turn, tagged with the stage that produced it."""

    idx: int
    """1-based position. Rendered as `[M1]` for judges to cite."""

    role: Role
    """`system` / `user` (auditor) / `assistant` (target)."""

    content: str
    """Verbatim text."""

    stage: Stage = ""
    """Stage that produced this turn, for slicing the transcript by stage."""

    @property
    def mid(self) -> str:
        """Citation id, e.g. `M3`."""
        return f"M{self.idx}"


@dataclass(frozen=True)
class LeakReport:
    """Result of checking that no index content reached the target too early."""

    ok: bool
    """True when nothing leaked."""

    violations: tuple[str, ...] = ()
    """One line per leak, naming the message and the offending string."""

    def __str__(self) -> str:
        if self.ok:
            return "leak check: clean"
        return "leak check FAILED:\n  " + "\n  ".join(self.violations)


@dataclass
class Transcript:
    """Append-only conversation between the auditor and the target."""

    messages: list[Message] = field(default_factory=list)
    """Every turn, in order."""

    def say(self, content: str, *, stage: Stage = "") -> Message:
        """Record an auditor turn (what the target is about to be shown)."""
        return self._append("user", content, stage)

    def heard(self, content: str, *, stage: Stage = "") -> Message:
        """Record the target's reply."""
        return self._append("assistant", content, stage)

    def system(self, content: str, *, stage: Stage = "") -> Message:
        """Record the persona induction, if the route uses a system prompt."""
        return self._append("system", content, stage)

    def _append(self, role: Role, content: str, stage: Stage) -> Message:
        msg = Message(idx=len(self.messages) + 1, role=role, content=content, stage=stage)
        self.messages.append(msg)
        return msg

    # ---- views ----

    def in_stage(self, *stages: Stage) -> list[Message]:
        """Messages produced by any of `stages`, in order."""
        wanted = set(stages)
        return [m for m in self.messages if m.stage in wanted]

    def before_stage(self, stage: Stage) -> list[Message]:
        """Messages recorded before `stage` first appears.

        The whole transcript if the stage never ran — the conservative reading
        for a leak check.
        """
        for i, m in enumerate(self.messages):
            if m.stage == stage:
                return self.messages[:i]
        return list(self.messages)

    def target_turns(self) -> list[Message]:
        """Just the target's replies."""
        return [m for m in self.messages if m.role == "assistant"]

    def to_messages(self) -> list[dict[str, str]]:
        """Provider-shaped history: `[{"role": ..., "content": ...}, ...]`."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    # ---- rendering ----

    def render(self, *, messages: Optional[Iterable[Message]] = None) -> str:
        """Judge-facing rendering with citable ids.

        Roles are named for what they are in this design — the auditor is an
        interviewer, not a user — so a judge is not misled into scoring the
        auditor's turns as user behaviour.
        """
        names = {"system": "SYSTEM", "user": "AUDITOR", "assistant": "TARGET"}
        return "\n\n".join(
            f"[{m.mid}] {names[m.role]}: {m.content}"
            for m in (self.messages if messages is None else messages)
        )

    # ---- the invariant ----

    def leak_check(
        self,
        forbidden: Iterable[str],
        *,
        before: Stage,
        min_len: int = 24,
    ) -> LeakReport:
        """Assert no `forbidden` string reached the target before `before`.

        `forbidden` is index content — item text and group names. Strings
        shorter than `min_len` are skipped: a group name like "philosophy" is a
        common English word and would false-positive on any message mentioning
        it, so short names are checked by the caller as whole words instead.
        """
        needles = [s for s in forbidden if len(s) >= min_len]
        violations: list[str] = []
        for msg in self.before_stage(before):
            if msg.role == "assistant":
                continue  # the target's own words cannot be a leak into it
            haystack = msg.content.lower()
            for needle in needles:
                if needle.lower() in haystack:
                    violations.append(
                        f"[{msg.mid}] stage={msg.stage!r} contains "
                        f"index text: {needle[:70]!r}"
                    )
        return LeakReport(ok=not violations, violations=tuple(violations))

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self) -> Iterator[Message]:
        return iter(self.messages)
