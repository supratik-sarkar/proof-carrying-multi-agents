"""
Cryptographic commitments (Definition 2.2, Eq. 2).

Implements:
    - H(.): canonical hash function for evidence payloads
    - Merkle root aggregation (Appendix A.1, Eq. 27) for tamper-evident audit logs
    - Utility helpers used by the checker to re-verify h(v) = H(x(v))

Design choices:
    - Default hash is SHA-256 via `hashlib` stdlib. `xxhash` is exposed for hot paths
      (overhead meter) where a non-cryptographic hash is acceptable; it is NEVER
      used for `h(v)` which must be collision-resistant.
    - Canonicalization is baked into the `GraphNode.content_for_hash()` method so
      this module does not need to know node internals.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable

_HASH_ALG = "sha256"


def H(data: bytes) -> str:
    """The commitment function H(.) from Eq. (2). Returns a hex digest.

    We return hex (not raw bytes) so that the digest can be stored inside
    dataclasses without needing base64 encoding for serialization.
    """
    return hashlib.new(_HASH_ALG, data).hexdigest()


def verify(data: bytes, expected_digest: str) -> bool:
    """Check that H(data) == expected_digest. Used by Check_clm (Eq. 13)."""
    return H(data) == expected_digest


# -----------------------------------------------------------------------------
# Merkle tree for the tamper-evident log (Appendix A.1, Eq. 27)
# -----------------------------------------------------------------------------


def merkle_root(leaves: Iterable[str]) -> str:
    """Compute the Merkle root over a list of hex-digest leaves.

    We use a binary Merkle tree with duplication of the last leaf for odd levels
    (the standard Bitcoin-style construction). Leaves are treated as hex digests
    and concatenated (bytes-level) before hashing.

    Returns the all-zeros digest of length 64 if the input is empty, which is a
    deterministic sentinel distinguishable from any real payload (H applied to a
    non-empty byte string will never produce all zeros with negligible probability).
    """
    current: list[bytes] = [bytes.fromhex(leaf) for leaf in leaves]
    if not current:
        return "0" * 64
    while len(current) > 1:
        if len(current) % 2 == 1:
            current.append(current[-1])
        next_level: list[bytes] = []
        for i in range(0, len(current), 2):
            next_level.append(hashlib.new(_HASH_ALG, current[i] + current[i + 1]).digest())
        current = next_level
    return current[0].hex()


@dataclass
class AuditLog:
    """Tamper-evident append-only log (Appendix A.1).

    Each append records a Merkle root over the current set of leaves, then the
    cumulative chain is computed as R_t = H(R_{t-1} || r_t). This gives a
    lightweight blockchain-style audit trail without adding a dependency on an
    actual distributed ledger.

    Used by the Verifier agent as a sanity check during R1 tamper detection.
    """

    leaves: list[str] = field(default_factory=list)      # hex digests h(v)
    roots: list[str] = field(default_factory=list)       # r_t at each step
    chain: list[str] = field(default_factory=list)       # R_t chained roots

    def append(self, leaf_digest: str) -> str:
        """Append a leaf and return the updated chained root R_t."""
        self.leaves.append(leaf_digest)
        r_t = merkle_root(self.leaves)
        self.roots.append(r_t)
        if not self.chain:
            R_t = H(r_t.encode("utf-8"))
        else:
            R_t = H((self.chain[-1] + r_t).encode("utf-8"))
        self.chain.append(R_t)
        return R_t

    @property
    def current_root(self) -> str:
        return self.chain[-1] if self.chain else "0" * 64

    def verify_prefix(self, other: "AuditLog") -> bool:
        """True iff `other` is a valid prefix of this log.

        Used by the Verifier to assert that no leaves have been silently deleted
        or re-ordered between prover and verifier.
        """
        n = len(other.leaves)
        if n > len(self.leaves):
            return False
        return all(self.leaves[i] == other.leaves[i] for i in range(n))
