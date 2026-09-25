"""Execution modes (D-M4). Resume must never masquerade as replication."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol

from ..canon import sha256_obj


class ExecutionMode(str, Enum):
    FRESH = "FRESH"          # no existing lineage may be reused
    RESUME = "RESUME"        # continue a known interrupted execution
    REPLICATE = "REPLICATE"  # same frozen spec, NEW identity, COLD lineage
    REPLAY = "REPLAY"        # from a named checkpoint under an explicit intervention


#: Only these are admissible as independent reproduction evidence.
REPRODUCTION_ADMISSIBLE = frozenset({ExecutionMode.FRESH, ExecutionMode.REPLICATE})


class ThreadCollision(RuntimeError):
    """A deterministic thread_id already exists and RESUME was not requested."""


@dataclass(frozen=True)
class ExecutionIdentity:
    run_id: str
    cell_id: str
    example_id: str
    system: str
    seed: int
    spec_hash: str
    mode: ExecutionMode

    @property
    def thread_id(self) -> str:
        """Deterministic over the LOGICAL identity.

        REPLICATE deliberately includes run_id so a fresh run identity yields a
        fresh thread and cannot resume a warm store.
        """
        base = {"cell_id": self.cell_id, "example_id": self.example_id,
                "system": self.system, "seed": self.seed, "spec_hash": self.spec_hash}
        if self.mode is ExecutionMode.REPLICATE:
            base["run_id"] = self.run_id
        return "th-" + sha256_obj(base)[:32]

    @property
    def admissible_for_reproduction(self) -> bool:
        return self.mode in REPRODUCTION_ADMISSIBLE


class CheckpointStoreLike(Protocol):
    def has_thread(self, thread_id: str) -> bool: ...


def open_execution(identity: ExecutionIdentity, store: CheckpointStoreLike) -> str:
    """Resolve a thread, enforcing mode semantics. Fails closed on collision."""
    tid = identity.thread_id
    warm = store.has_thread(tid)
    if identity.mode is ExecutionMode.FRESH and warm:
        raise ThreadCollision(
            f"thread {tid} already exists; FRESH may not reuse existing lineage. "
            "Select RESUME explicitly, or REPLICATE for independent reproduction.")
    if identity.mode is ExecutionMode.RESUME and not warm:
        raise ThreadCollision(f"thread {tid} has no lineage to resume")
    if identity.mode is ExecutionMode.REPLICATE and warm:
        # Cannot happen while run_id is in the digest; assert it loudly if it does.
        raise ThreadCollision(
            f"REPLICATE resolved to a warm thread {tid}: replication would have "
            "silently become resume.")
    return tid
