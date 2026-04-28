"""
Bring-your-own-key sidebar.

Renders a provider selector + (when relevant) a password input for the
user's API key. Keys live ONLY in `st.session_state["api_keys"][provider]`
— never written to disk, never logged, never sent anywhere except the
provider's own endpoint.

The selector is the same on every page so users don't have to re-pick.
"""
from __future__ import annotations

import streamlit as st

from .llm_client import PROVIDERS, free_provider_info, list_premium_providers


def _ensure_state() -> None:
    """Make sure session_state has the slots we use."""
    if "api_keys" not in st.session_state:
        st.session_state["api_keys"] = {}   # provider_id -> key
    if "provider_id" not in st.session_state:
        st.session_state["provider_id"] = "hf_free"
    if "model_id" not in st.session_state:
        st.session_state["model_id"] = free_provider_info().default_model


def render_byok_sidebar() -> tuple[str, str | None, str]:
    """Sidebar widget. Returns (provider_id, api_key, model_id).

    The returned api_key is only non-None for premium providers; free
    providers always get None and the LLMClient handles the optional
    server-side HF_TOKEN automatically.
    """
    _ensure_state()

    with st.sidebar:
        st.markdown("### Backend")
        st.caption("Default is free. Add an API key to unlock premium models.")

        # Provider selector
        labels = [p.label for p in PROVIDERS.values()]
        ids = list(PROVIDERS.keys())
        cur_index = (
            ids.index(st.session_state["provider_id"])
            if st.session_state["provider_id"] in ids else 0
        )
        chosen_label = st.selectbox(
            "Provider",
            labels,
            index=cur_index,
            help=(
                "HF Inference (free tier) is the default and costs you "
                "nothing. Premium providers require your own API key."
            ),
        )
        provider_id = ids[labels.index(chosen_label)]
        st.session_state["provider_id"] = provider_id
        info = PROVIDERS[provider_id]

        # Model selector
        cur_model = st.session_state["model_id"]
        if cur_model not in info.available_models:
            cur_model = info.default_model
        model_id = st.selectbox(
            "Model",
            info.available_models,
            index=info.available_models.index(cur_model),
        )
        st.session_state["model_id"] = model_id

        # API key (only for premium providers)
        api_key = None
        if info.requires_key:
            saved = st.session_state["api_keys"].get(provider_id, "")
            api_key = st.text_input(
                f"{info.label} API key",
                value=saved,
                type="password",
                placeholder="paste here — kept in browser session only",
                help=(
                    f"Get one at {info.docs_url}. Never logged or persisted; "
                    "wiped when you close the tab."
                ),
            )
            st.session_state["api_keys"][provider_id] = api_key
            if not api_key:
                st.warning(
                    f"This provider requires a key. Without one, runs will "
                    f"fall back to {free_provider_info().label}."
                )
        else:
            # The default `hf_free` provider uses the Space-side HF_TOKEN
            # (set by the Space owner as a secret). If it isn't set, every
            # call returns 401 — be honest about that.
            import os as _os
            if _os.environ.get("HF_TOKEN"):
                st.success(
                    "Default backend ready (Space-side HF token configured)."
                )
            else:
                st.warning(
                    "No HF token detected on the Space. Live runs may fail "
                    "with 401. Add a free HF account token in the sidebar "
                    "above (HF Inference, your token) to enable Live Run."
                )

        st.markdown("---")
        # ----- Agentic mode toggle -----
        # The PCG-MAS framework treats single-agent as the trivial special
        # case k=1 of the redundant-consensus law (Theorem 2). We expose
        # both modes so reviewers can flip between them and see why
        # multi-agent is the meaningful general form.
        st.markdown("### Agentic mode")
        mode = st.radio(
            "Pipeline shape",
            options=["Single-agent (k=1)", "Multi-agent (k=2)", "Multi-agent (k=4)"],
            index=st.session_state.get("agentic_mode_idx", 1),
            help=(
                "Single-agent runs one Prover. By Theorem 2, this is the "
                "k=1 special case of the redundant-consensus law: the "
                "false-accept bound is ε (no consensus collapse). "
                "Multi-agent runs k Provers and applies consensus, so the "
                "false-accept bound shrinks geometrically as ρ^(k−1)·ε^k. "
                "PCG-MAS works in either mode; multi-agent is just the "
                "general case."
            ),
        )
        st.session_state["agentic_mode_idx"] = (
            ["Single-agent (k=1)", "Multi-agent (k=2)", "Multi-agent (k=4)"]
            .index(mode)
        )
        # Numeric k for the run scripts
        k_map = {
            "Single-agent (k=1)": 1,
            "Multi-agent (k=2)": 2,
            "Multi-agent (k=4)": 4,
        }
        st.session_state["k_redundant"] = k_map[mode]
        st.session_state["agentic_mode"] = mode

        # Brief unification note (the why)
        if "Single-agent" in mode:
            st.caption(
                "Single-agent is the trivial k=1 special case of PCG-MAS. "
                "The certificate Z is still produced and verifiable, "
                "but with no redundancy collapse."
            )
        else:
            st.caption(
                "Multi-agent activates Theorem 2's geometric collapse: "
                "k Provers in parallel + verifier consensus → bound "
                "shrinks as ρ^(k−1)·ε^k."
            )

        # Privacy note (always shown)
        with st.expander("Privacy & cost", expanded=False):
            st.markdown(
                "**Cost.** The default `HF Inference (free tier)` is rate-"
                "limited but free for everyone. Premium providers charge "
                "you directly — typical demo run costs $0.001 – $0.05.\n\n"
                "**Privacy.** API keys live only in `st.session_state` "
                "(your browser tab). They are NOT sent to this server, "
                "NOT logged, and disappear when you close the tab.\n\n"
                "**Anonymity.** This demo is hosted anonymously for "
                "double-blind review. We do not run analytics."
            )

    return provider_id, (api_key or None), model_id


def get_active_client():
    """Convenience: return an LLMClient configured from current session state.

    Falls back to the free provider when a premium key is missing rather
    than crashing the page mid-render."""
    from .llm_client import LLMClient, free_provider_info, PROVIDERS

    _ensure_state()
    provider_id = st.session_state.get("provider_id", "hf_free")
    info = PROVIDERS.get(provider_id, free_provider_info())
    api_key = st.session_state.get("api_keys", {}).get(provider_id) or None
    model_id = st.session_state.get("model_id") or info.default_model

    if info.requires_key and not api_key:
        # graceful fallback
        st.info(
            f"No API key for {info.label}; falling back to "
            f"{free_provider_info().label}."
        )
        return LLMClient(provider="hf_free", model=free_provider_info().default_model)

    return LLMClient(provider=provider_id, api_key=api_key, model=model_id)
