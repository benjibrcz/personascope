"""Provider wrapper that injects YAWYR's TAG_PREFIX into probe questions.

The tagged SFT models (`*-tagged-*`) are trained to fire the persona only
when the user prompt contains the YAWYR format instruction:

    You will be asked a question. Always reply in the format:

    <START> "your answer here" <END>

…immediately followed by the question. To avoid modifying every probe
builder to thread a "question prefix" parameter through, this wrapper
sits in front of any UnifiedProvider-shaped object and prepends
TAG_PREFIX to the *last* user message of every `complete()` call. The
ICL prefix (built separately by `sample_tagged_icl_context`) is left
unchanged.

Usage:
    base = get_provider("ft-stalin-tagged")
    p = TaggedEvalProvider(base)
    p.complete(messages=...)  # last user message gets TAG_PREFIX prepended
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from personascope.core.runner import TAG_PREFIX


class TaggedEvalProvider:
    """Lightweight wrapper that retags the eval question.

    Forwards every `complete()` call to the underlying provider after
    prepending TAG_PREFIX to the final user message in the messages
    list. Pass-throughs:
      - .config → underlying config (for caching keys, name, etc.)
      - .name, .model → fall through via attribute lookup
    """

    def __init__(self, inner):
        self._inner = inner

    @property
    def config(self):
        return getattr(self._inner, "config", None)

    @property
    def name(self) -> str:
        cfg = self.config
        base = getattr(cfg, "name", "") if cfg else ""
        return f"TaggedEval({base})" if base else "TaggedEval"

    @property
    def model(self) -> str:
        cfg = self.config
        return getattr(cfg, "model", "") if cfg else ""

    @staticmethod
    def _retag(messages: list[dict]) -> list[dict]:
        """Return a new list with TAG_PREFIX prepended to the last user message."""
        out = [deepcopy(m) for m in messages]
        # Find the last user message
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") == "user":
                content = out[i].get("content") or ""
                if not content.startswith("You will be asked"):
                    out[i]["content"] = TAG_PREFIX + content
                break
        return out

    def complete(self, messages: list[dict], **kwargs: Any) -> Any:
        retagged = self._retag(messages)
        return self._inner.complete(messages=retagged, **kwargs)


__all__ = ["TaggedEvalProvider"]
