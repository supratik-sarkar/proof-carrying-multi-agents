"""Explicit pinned-NLI binding. Fails closed.

verifier.DEFAULT_SNAPSHOT_DIR points at ~/.cache/huggingface/hub, which does
not exist on the execution machine. The real 1.7 GB snapshot lives in the
repository. Never let the default be selected.
"""
from pathlib import Path
from typing import Any, Dict

from pcg.v3_5.verifier import PINNED_MODEL_ID, PINNED_REVISION, PinnedDeBERTaVerifier

REPO_SNAPSHOT_RELPATH = "artifacts/v3_6/route_b_v2_2_2/checkpoints/pcg_nli_deberta_v3_large"
REQUIRED_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")


class NLIBindingError(RuntimeError):
    pass


def snapshot_path(repo_root) -> Path:
    from .authority_resolver import resolve_path
    return resolve_path(Path(repo_root).resolve() / REPO_SNAPSHOT_RELPATH)


def assert_snapshot(repo_root) -> Path:
    p = snapshot_path(repo_root)
    if not p.is_dir():
        raise NLIBindingError(f"PINNED_NLI_SNAPSHOT_MISSING:{p}")
    missing = [f for f in REQUIRED_FILES if not (p / f).is_file()]
    if missing:
        raise NLIBindingError(f"PINNED_NLI_SNAPSHOT_INCOMPLETE:{missing}")
    return p


def bind_verifier(repo_root) -> PinnedDeBERTaVerifier:
    """Construct the pinned verifier against the repository snapshot only."""
    p = assert_snapshot(repo_root)
    v = PinnedDeBERTaVerifier(snapshot_path=p)
    prov: Dict[str, Any] = v.get_provenance() or {}
    rev = prov.get("revision") or prov.get("pinned_revision")
    mid = prov.get("model_id") or prov.get("pinned_model_id")
    if rev is not None and rev != PINNED_REVISION:
        raise NLIBindingError(f"PINNED_NLI_REVISION_MISMATCH:expected={PINNED_REVISION}:got={rev}")
    if mid is not None and mid != PINNED_MODEL_ID:
        raise NLIBindingError(f"PINNED_NLI_MODEL_MISMATCH:expected={PINNED_MODEL_ID}:got={mid}")
    return v


def binding_record(repo_root) -> Dict[str, Any]:
    p = assert_snapshot(repo_root)
    return {"pinned_model_id": PINNED_MODEL_ID, "pinned_revision": PINNED_REVISION,
            "snapshot_path": str(p), "source": "REPOSITORY_LOCAL_SNAPSHOT",
            "default_hf_cache_used": False}
