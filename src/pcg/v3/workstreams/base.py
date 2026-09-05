"""Workstream base: frozen spec, artifact contract, idempotent offline runner."""
from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..canon import canonical_json, sha256_file, sha256_obj
from ..release import ARTIFACT_SCHEMA_VERSION, METRIC_VERSION, PCG_MAS_RELEASE

ARTIFACT_ROOT = os.environ.get("PCG_ARTIFACT_ROOT", "artifacts/v3_0")
REQUIRED_FILES = ("README.md", "spec.json", "environment.json", "metrics.json",
                  "RESULT.md", "checks.json", "SHA256SUMS")


@dataclass
class Spec:
    """Frozen pre-registration. Evaluation FAILS if the hash changes."""
    experiment_id: str
    name: str
    provenance_class: str
    requires_model_calls: bool
    tier: str
    params: Dict[str, Any] = field(default_factory=dict)
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    metric_version: str = METRIC_VERSION
    release: str = PCG_MAS_RELEASE

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}

    @property
    def spec_hash(self) -> str:
        return sha256_obj(self.to_dict())


def environment_payload() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "release": PCG_MAS_RELEASE,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "device": _device(),
        "captured_utc": datetime.now(timezone.utc).isoformat(),
    }


def _device() -> str:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class Workstream:
    """Base class. `run()` must be offline and finite in this release pass."""

    spec: Spec

    def __init__(self, spec: Spec, root: str = ARTIFACT_ROOT):
        self.spec = spec
        self.root = root

    # -- paths -------------------------------------------------------------
    @property
    def outdir(self) -> str:
        return os.path.join(self.root, self.spec.experiment_id.lower())

    def fixture_path(self) -> str:
        return os.path.join("tests", "fixtures", "v3", self.spec.experiment_id.lower(), "records.jsonl")

    # -- data --------------------------------------------------------------
    def load_records(self, path: Optional[str] = None) -> List[dict]:
        p = path or self.fixture_path()
        if not os.path.exists(p):
            return []
        out = []
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    # -- to implement ------------------------------------------------------
    def compute(self, records: List[dict]) -> Dict[str, Any]:
        raise NotImplementedError

    def checks(self, metrics: Dict[str, Any], records: List[dict]) -> Dict[str, Any]:
        return {"records_present": len(records) > 0}

    def result_markdown(self, metrics: Dict[str, Any], checks: Dict[str, Any]) -> str:
        lines = [f"# {self.spec.experiment_id} — {self.spec.name}", "",
                 f"- release: `{PCG_MAS_RELEASE}`",
                 f"- provenance class: `{self.spec.provenance_class}`",
                 f"- spec hash: `{self.spec.spec_hash[:16]}`",
                 f"- requires model calls: `{self.spec.requires_model_calls}`", "",
                 "## Metrics", "```json", canonical_json(metrics)[:4000], "```", "",
                 "## Checks", "```json", canonical_json(checks), "```", "",
                 "## Failures and limitations", ""]
        failed = [k for k, v in checks.items() if v is False]
        lines.append("None recorded." if not failed else
                     "\n".join(f"- FAILED check: `{k}`" for k in failed))
        return "\n".join(lines) + "\n"

    # -- execution ---------------------------------------------------------
    def run(self, records: Optional[List[dict]] = None) -> Dict[str, Any]:
        recs = records if records is not None else self.load_records()
        os.makedirs(self.outdir, exist_ok=True)
        metrics = self.compute(recs)
        metrics.setdefault("experiment_id", self.spec.experiment_id)
        metrics.setdefault("metric_version", METRIC_VERSION)
        metrics.setdefault("spec_hash", self.spec.spec_hash)
        metrics.setdefault("n_records", len(recs))
        chk = self.checks(metrics, recs)

        self._write("spec.json", canonical_json(self.spec.to_dict()))
        self._write("environment.json", canonical_json(environment_payload()))
        self._write("metrics.json", canonical_json(metrics))
        self._write("checks.json", canonical_json(chk))
        self._write("RESULT.md", self.result_markdown(metrics, chk))
        self._write("README.md",
                    f"# {self.spec.experiment_id} — {self.spec.name}\n\n"
                    f"Reproduce:\n\n    python -m pcg.v3.workstreams.cli run {self.spec.experiment_id}\n\n"
                    f"Offline; no model or network call. Provenance class "
                    f"`{self.spec.provenance_class}`.\n")
        self._sha256sums()
        return {"metrics": metrics, "checks": chk, "outdir": self.outdir}

    def _write(self, name: str, text: str) -> None:
        with open(os.path.join(self.outdir, name), "w") as fh:
            fh.write(text if text.endswith("\n") else text + "\n")

    def _sha256sums(self) -> None:
        rows = []
        for fn in sorted(os.listdir(self.outdir)):
            if fn == "SHA256SUMS":
                continue
            fp = os.path.join(self.outdir, fn)
            if os.path.isfile(fp):
                rows.append(f"{sha256_file(fp)}  {fn}")
        self._write("SHA256SUMS", "\n".join(rows))


def verify_spec(spec: Spec, frozen_hash: Optional[str]) -> None:
    """Explicit failure on spec drift. Never a permissive fallback."""
    if frozen_hash and spec.spec_hash != frozen_hash:
        raise RuntimeError(
            f"frozen spec drift for {spec.experiment_id}: {spec.spec_hash[:16]} != {frozen_hash[:16]}. "
            "Re-freezing after observing results is prohibited.")
