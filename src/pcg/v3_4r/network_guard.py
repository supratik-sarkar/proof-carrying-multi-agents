"""PCG-MAS v3.4R Hard Network & Provider Guard.

Enforces:
- Hard socket connection blocking (socket.socket.connect, socket.create_connection)
- Recording and auditing of all attempted network operations
- Stripping of all provider credentials from environment variables
- Verification that NETWORK_ATTEMPTS == 0
"""

import os
import socket
from typing import Any, Dict, List, Optional, Tuple

NETWORK_ATTEMPT_LOG: List[Dict[str, Any]] = []

PROVIDER_ENV_PATTERNS = [
    "OPENAI",
    "ANTHROPIC",
    "GEMINI",
    "MISTRAL",
    "COHERE",
    "HUGGINGFACE",
    "HF_",
    "WANDB",
    "AWS_",
    "AZURE_",
]


def strip_provider_credentials() -> List[str]:
    """Removes all provider API keys and tokens from os.environ."""
    removed = []
    for k in list(os.environ.keys()):
        upper_k = k.upper()
        if any(p in upper_k for p in PROVIDER_ENV_PATTERNS) or any(
            t in upper_k for t in ["KEY", "TOKEN", "SECRET", "AUTH"]
        ):
            # Keep python/system internals intact
            if k in (
                "PYTHONHASHSEED",
                "PYTHONPATH",
                "PATH",
                "USER",
                "HOME",
                "SHELL",
            ):
                continue
            del os.environ[k]
            removed.append(k)
    return removed


def install_network_block() -> None:
    """Installs monkey-patches on socket to block all outgoing network connections."""
    orig_connect = socket.socket.connect

    def blocked_connect(self, address, *args, **kwargs):
        NETWORK_ATTEMPT_LOG.append(
            {
                "target": str(address),
                "blocked": True,
            }
        )
        raise RuntimeError(
            f"NETWORK BLOCK ACTIVATED: Outgoing connection to {address} is strictly forbidden in v3.4R!"
        )

    socket.socket.connect = blocked_connect

    # Also block create_connection if present
    if hasattr(socket, "create_connection"):

        def blocked_create_connection(address, *args, **kwargs):
            NETWORK_ATTEMPT_LOG.append(
                {
                    "target": str(address),
                    "blocked": True,
                }
            )
            raise RuntimeError(
                f"NETWORK BLOCK ACTIVATED: socket.create_connection to {address} forbidden!"
            )

        socket.create_connection = blocked_create_connection


def get_network_audit_report() -> Dict[str, Any]:
    """Returns network attempt audit summary."""
    return {
        "network_attempts": len(NETWORK_ATTEMPT_LOG),
        "attempts_log": list(NETWORK_ATTEMPT_LOG),
        "network_blocked": True,
        "clean": len(NETWORK_ATTEMPT_LOG) == 0,
    }
