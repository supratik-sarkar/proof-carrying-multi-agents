"""
Hugging Face authentication helpers.

Tokens are never read from source-controlled files and are never printed.
Allowed runtime sources:
  1. explicit argument
  2. HF_INFERENCE
  3. HF_HUB_READ
  4. HF_TOKEN
  5. HUGGINGFACE_HUB_TOKEN
  4. interactive terminal / notebook input
"""
from __future__ import annotations

import getpass
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class HFAuthResult:
    token: str | None
    source: str
    full_access: bool
    message: str


def resolve_hf_token(
    *,
    explicit_token: str | None = None,
    require_for_full: bool = False,
    interactive: bool = True,
    prompt: str = "Enter Hugging Face token for full model rerun, or press Enter to use offline fallback: ",
) -> HFAuthResult:
    """Resolve an HF token without leaking it.

    If token is missing:
      - require_for_full=True returns full_access=False with a clear message.
      - interactive=True prompts the user securely.
      - blank input selects offline/preflight fallback.
    """
    if explicit_token and explicit_token.strip():
        return HFAuthResult(
            token=explicit_token.strip(),
            source="explicit",
            full_access=True,
            message="Using HF token supplied explicitly.",
        )

    for env_name in ("HF_INFERENCE", "HF_HUB_READ", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(env_name)
        if token and token.strip():
            return HFAuthResult(
                token=token.strip(),
                source=env_name,
                full_access=True,
                message=f"Using HF token from {env_name}.",
            )

    if interactive:
        try:
            token = getpass.getpass(prompt)
        except (EOFError, KeyboardInterrupt):
            token = ""

        if token and token.strip():
            os.environ["HF_TOKEN"] = token.strip()
            return HFAuthResult(
                token=token.strip(),
                source="interactive",
                full_access=True,
                message="Using HF token entered interactively for this process only.",
            )

    msg = (
        "You entered a null HF token. Running the feasible offline/preflight path "
        "instead of the full model rerun. Remote/gated HF model cells will be skipped "
        "unless HF_TOKEN or HUGGINGFACE_HUB_TOKEN is set."
    )

    if require_for_full:
        return HFAuthResult(
            token=None,
            source="missing",
            full_access=False,
            message=msg,
        )

    return HFAuthResult(
        token=None,
        source="missing",
        full_access=False,
        message=msg,
    )


def require_hf_token_for_remote_backend(explicit_token: str | None = None) -> str:
    """Return token or raise a clean error for remote-only code paths."""
    auth = resolve_hf_token(
        explicit_token=explicit_token,
        require_for_full=True,
        interactive=False,
    )
    if auth.token:
        return auth.token

    raise RuntimeError(
        "HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required for HFInferenceBackend. "
        "Set one in the environment or use the offline MockBackend/preflight mode."
    )
