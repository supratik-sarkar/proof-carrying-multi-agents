"""Canonical record identity.

`model|dataset|condition|seed|example` is human-readable and insufficient: it
omits provider, backend, experiment config and decoding config, so two records
from different configurations collide. Identity here is a structured object,
hashed canonically; `record_id` is derived from the hash, never concatenated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from .hashing import hash_obj

IDENTITY_VERSION = "pcg-identity/1"


@dataclass(frozen=True)
class RecordIdentity:
    experiment_id: str
    dataset: str
    example_id: str
    condition: str
    seed: int
    provider: str
    backend: str
    requested_model: str
    experiment_config_hash: str
    decoding_config_hash: str
    input_hash: str                  # hash of the canonical prompt/input
    identity_version: str = IDENTITY_VERSION

    def canonical(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        return hash_obj(self.canonical())

    def record_id(self) -> str:
        return f"pcgrec_{self.digest()[:32]}"
