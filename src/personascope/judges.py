"""LLM judges, in one place.

A judge is a `(prompt) -> text` callable plus the upstream id it resolved to.
Both halves matter: the alias is what a config names, the resolved id is what
actually answered and is what a manifest has to record.

These specs do not go through `models.resolve_model`. That function resolves
*targets* — the models under measurement, pinned to one serving stack by
`configs/models.yaml` so a number is reproducible. Judges are instruments, not
subjects: they always run on the vendor's own API at temperature 0, and they
are versioned by the rubric they read rather than by the model grid. Keeping
them apart also means a change to the model grid cannot silently re-point a
judge and orphan a parsed file.

Changing an entry here changes every `parse_key` that names it, which forces a
re-parse (never a re-ask — judges run in the parse pass). Add entries; edit
them only deliberately.
"""
from __future__ import annotations

from typing import Any, Callable

__all__ = ["JUDGES", "judge_fn", "resolved_id"]

JUDGES: dict[str, dict[str, Any]] = {
    # OpenAI's own API; the 5.x line takes max_completion_tokens and no temperature.
    #
    # `reasoning_effort` is "low", not "minimal": at minimal, gpt-5-mini answers
    # NO to every identity rubric it is shown (checked 2026-09-18).
    #
    # The dated snapshot, not the `gpt-5-mini` alias. An alias is repointed by
    # the vendor without notice, and a judge that changes under a finished run
    # makes its numbers unreproducible while every file still says `gpt-5-mini`.
    # Pinned 2026-09-22; the alias resolved here at the time, so verdicts do
    # not move, only the parse_key that names them.
    "gpt-5-mini": {"model": "gpt-5-mini-2025-08-07", "api_key_env": "OPENAI_API_KEY",
                   "max_completion_tokens_param": True, "send_temperature": False,
                   "reasoning_effort": "low", "max_tokens": 400},
    # The dated snapshot, not the `gpt-5` alias, which moves under us.
    #
    # The cap is generous because it is shared with the reasoning trace: at 600,
    # 9 of 1148 SAD labelling calls spent the whole budget reasoning and returned
    # empty content, which reads downstream as a judge failure rather than as
    # what it is.
    "gpt-5": {"model": "gpt-5-2025-08-07", "api_key_env": "OPENAI_API_KEY",
              "max_completion_tokens_param": True, "send_temperature": False,
              "reasoning_effort": "low", "max_tokens": 2000},
    "gpt-4.1": {"model": "gpt-4.1-2025-04-14", "api_key_env": "OPENAI_API_KEY",
                "max_tokens": 16},
}


def resolved_id(name: str) -> str:
    """The upstream model string judge `name` sends as `model=`."""
    return JUDGES[name]["model"]


def judge_fn(name: str, *, max_tokens: int | None = None) -> Callable[[str], str]:
    """A `(prompt) -> text` judge.

    Raises on an upstream failure rather than returning empty text: a judge
    that fails quietly turns into a label, and a label that is really an
    outage is indistinguishable from one the model meant.
    """
    from personascope.llm.provider import ProviderConfig, UnifiedProvider

    spec = dict(JUDGES[name])
    cap = max_tokens if max_tokens is not None else spec.pop("max_tokens")
    spec.pop("max_tokens", None)
    provider = UnifiedProvider(ProviderConfig(name=f"judge:{name}", **spec))

    def judge(prompt: str) -> str:
        res = provider.complete(messages=[{"role": "user", "content": prompt}],
                                max_tokens=cap, temperature=0.0)
        if not res.get("success", True):
            raise RuntimeError(f"judge {name}: {res.get('error')}")
        return (res.get("text") or "").strip()

    return judge
