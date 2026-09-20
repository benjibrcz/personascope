"""Provider calls: one place that talks to an API.

Split out so the runner, the judge and any later analysis share one retry
policy and one place to change routing. Defaults to OpenRouter per
`configs/models.yaml`; `--provider openai` switches to OpenAI direct, which the
self-report pilot used.

A transport failure raises. It is never recorded as an empty answer, because an
API error scored as model behaviour is indistinguishable from a refusal in the
results and would quietly become a finding.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

__all__ = ["Client", "ProviderError", "ROUTES"]


class ProviderError(RuntimeError):
    """A call failed after exhausting retries."""


ROUTES = {
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
}


@dataclass
class Client:
    """Minimal chat-completions client with bounded retries."""

    model: str
    provider: str = "openrouter"
    temperature: float = 1.0
    max_tokens: int = 512
    seed: Optional[int] = 42
    timeout: int = 90
    retries: int = 4

    def __post_init__(self) -> None:
        if self.provider not in ROUTES:
            raise ValueError(f"Unknown provider {self.provider!r}; use one of {list(ROUTES)}")
        self.url, key_env = ROUTES[self.provider]
        self.key = os.environ.get(key_env)
        if not self.key:
            raise RuntimeError(f"{key_env} not set (provider={self.provider})")

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if self.seed is not None:
            body["seed"] = self.seed

        payload = json.dumps(body).encode()
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}

        last: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.load(resp)
                return (data["choices"][0]["message"]["content"] or "").strip()
            except urllib.error.HTTPError as exc:
                last = exc
                # 4xx other than rate-limiting will not fix themselves.
                if exc.code not in (408, 409, 429) and exc.code < 500:
                    raise ProviderError(f"HTTP {exc.code}: {exc.read()[:300]!r}") from exc
            except Exception as exc:  # noqa: BLE001 - transport, retry
                last = exc
            time.sleep(2**attempt)

        raise ProviderError(f"failed after {self.retries} attempts: {last}")
