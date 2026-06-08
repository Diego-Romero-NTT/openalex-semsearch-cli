"""Use the operating system's trust store instead of the certifi bundle.

Required on corporate networks with TLS inspection (e.g. Zscaler), where the root
certificate lives in the OS keychain and not in certifi. `inject_into_ssl` makes
httpx and the OpenAI SDK verify against that trust store.
"""

from __future__ import annotations

import truststore

_injected = False


def ensure_system_trust() -> None:
    """Inject the OS trust store into `ssl` (idempotent)."""
    global _injected
    if not _injected:
        truststore.inject_into_ssl()
        _injected = True
