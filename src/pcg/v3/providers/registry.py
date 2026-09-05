"""Provider registry with a hard offline guard."""
from __future__ import annotations
import os
from typing import Any, Dict

from .offline_mock import OfflineMockProvider

OFFLINE_ONLY = os.environ.get("PCG_OFFLINE_ONLY", "1") == "1"


class NetworkCallBlocked(RuntimeError):
    pass


def get_provider(route: str = "offline_mock", **kw):
    if route == "offline_mock":
        return OfflineMockProvider(**kw)
    if OFFLINE_ONLY:
        raise NetworkCallBlocked(
            f"provider route {route!r} requires a model/network call but PCG_OFFLINE_ONLY=1. "
            "This guard exists so a remediation pass cannot silently spend budget.")
    if route == "local_hf":
        from .local_hf import LocalHFProvider   # noqa: F401  (lazy, optional dep)
        return LocalHFProvider(**kw)
    if route == "hosted_provider":
        from .hosted import HostedProvider      # noqa: F401
        return HostedProvider(**kw)
    raise ValueError(f"unknown provider route {route!r}")


def status() -> Dict[str, Any]:
    return {"offline_only": OFFLINE_ONLY, "available_routes": ["offline_mock"],
            "gated_routes": ["local_hf", "hosted_provider"]}
