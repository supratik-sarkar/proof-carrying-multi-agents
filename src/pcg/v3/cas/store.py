"""Content-addressed store. Verifiable, NOT trusted.

Every read rehashes the stored bytes and compares against the address, so a
corrupt or substituted object is detected rather than believed. The store adds
no new trusted component: `H` and `Canon` are already in the declared TCB.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from ..canon import canonical_json
from .address import Address, CHILD_ORDER, ObjType, addr
from .fields import committed_of


class CASCorruption(RuntimeError):
    pass


class CASMissing(KeyError):
    pass


class ContentAddressedStore:
    def __init__(self, root: Optional[str] = None):
        self.root = root
        self._mem: Dict[str, bytes] = {}
        if root:
            os.makedirs(root, exist_ok=True)

    # ------------------------------------------------------------ internals
    def _path(self, a: str) -> str:
        d = Address.parse(a).digest
        return os.path.join(self.root, d[:2], d[2:4], f"{d}.json")

    @staticmethod
    def _envelope(objtype: ObjType, committed: Mapping[str, Any],
                  children: Mapping[str, Sequence[Address]]) -> bytes:
        return canonical_json({
            "objtype": objtype.value,
            "committed": dict(committed),
            "children": {k: [str(x) for x in v] for k, v in (children or {}).items()},
        }).encode("utf-8")

    # ---------------------------------------------------------------- write
    def put(self, objtype: ObjType, payload: Mapping[str, Any],
            children: Optional[Mapping[str, Sequence[Address]]] = None) -> Address:
        """Store an object. Observed fields are stripped and never addressed."""
        committed = committed_of(payload)
        kids = {k: list(v or []) for k, v in (children or {}).items()}
        a = addr(objtype, committed, kids)
        blob = self._envelope(objtype, committed, kids)
        if self.root:
            p = self._path(a)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            if not os.path.exists(p):
                fd, tmp = tempfile.mkstemp(dir=os.path.dirname(p))
                with os.fdopen(fd, "wb") as fh:
                    fh.write(blob)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, p)          # atomic
        else:
            self._mem[str(a)] = blob
        return a

    # ----------------------------------------------------------------- read
    def has(self, a: str) -> bool:
        return os.path.exists(self._path(a)) if self.root else str(a) in self._mem

    def get(self, a: str) -> Dict[str, Any]:
        """Read AND verify. Corruption raises; it is never returned silently."""
        if self.root:
            p = self._path(a)
            if not os.path.exists(p):
                raise CASMissing(str(a))
            blob = open(p, "rb").read()
        else:
            if str(a) not in self._mem:
                raise CASMissing(str(a))
            blob = self._mem[str(a)]
        env = json.loads(blob.decode("utf-8"))
        recomputed = addr(ObjType(env["objtype"]), env["committed"],
                          {k: [Address(x) for x in v] for k, v in env["children"].items()})
        if str(recomputed) != str(a):
            raise CASCorruption(
                f"stored object does not hash to its address: {a} != {recomputed}")
        return env

    def verify(self, a: str) -> bool:
        try:
            self.get(a)
            return True
        except (CASCorruption, CASMissing):
            return False

    # ------------------------------------------------------------- traversal
    def children_of(self, a: str) -> List[Address]:
        env = self.get(a)
        out: List[Address] = []
        for slot in CHILD_ORDER.get(ObjType(env["objtype"]), ()):
            out.extend(Address(x) for x in env["children"].get(slot, []))
        return out

    def reachable(self, root: str) -> Set[str]:
        seen: Set[str] = set()
        stack = [str(root)]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            try:
                stack.extend(str(c) for c in self.children_of(cur))
            except CASMissing:
                pass
            except CASCorruption:
                pass
        return seen
