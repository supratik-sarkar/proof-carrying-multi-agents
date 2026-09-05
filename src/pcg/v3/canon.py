"""Canonical serialization and hashing.

Every hash in v3.0 is computed over this canonical form so that fingerprints are
stable across library versions, platforms and key insertion order.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

CANON_SEPARATORS = (",", ":")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, UTF-8 safe."""
    return json.dumps(obj, sort_keys=True, separators=CANON_SEPARATORS,
                      ensure_ascii=False, allow_nan=False, default=_default)


def _default(o: Any):
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    raise TypeError(f"not canonically serializable: {type(o)!r}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(**fields: Any) -> str:
    """Order-independent fingerprint over named fields (backend, checker, spec...)."""
    return sha256_obj(fields)
