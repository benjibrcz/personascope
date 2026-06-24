"""personascope.llm — self-contained LLM provider layer for personascope.

Small OpenAI-compatible interface that handles OpenAI direct + OpenRouter
with provider pinning. Add new providers in `provider.py`'s `PROVIDERS`
dict.
"""

from personascope.llm.provider import (
    PROVIDERS,
    ProviderConfig,
    UnifiedProvider,
    get_provider,
    list_providers,
)

__all__ = [
    "PROVIDERS",
    "ProviderConfig",
    "UnifiedProvider",
    "get_provider",
    "list_providers",
]
