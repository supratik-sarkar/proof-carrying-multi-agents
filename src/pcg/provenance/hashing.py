"""Canonical hashing. Every digest in the system is produced here."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

CANON_VERSION = "pcg-canon/2"


def canonical_text(s: str) -> bytes:
    s = unicodedata.normalize("NFC", s).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in s.split("\n")).encode("utf-8")


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str).encode("utf-8")


def hash_text(s: str) -> str:
    return hashlib.sha256(canonical_text(s)).hexdigest()


def hash_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hash_set(items) -> str:
    """Order-independent digest of a collection of hex digests."""
    return hashlib.sha256("".join(sorted(items)).encode("ascii")).hexdigest()
