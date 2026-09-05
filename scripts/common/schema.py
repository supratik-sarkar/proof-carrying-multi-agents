# scripts/common/schema.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import json
import hashlib

ALLOWED_PROVENANCE_VALUES: Set[str] = {"executed", "reimplementation", "cached", "unavailable"}

AUDIT_CHANNELS = ["Integrity", "Replay", "Drift", "Checker", "Coverage"]
SYSTEMS = [
    "No certificate",
    "ShieldAgent",
    "AgentRR",
    "VeriMAP",
    "PCG-MAS",
    "PCG-MAS:NoReplay",
    "PCG-MAS:NoRedundancy",
    "PCG-MAS:NoResp",
    "PCG-MAS:NoRiskCtrl",
]

def validate_record_provenance(record: Dict[str, Any], *, source_name: str = "metric_record") -> str:
    """Enforces Rule Zero Invariant on a single metric record.

    Rules:
    1. 'provenance' field is MANDATORY. Missing provenance fails schema validation.
    2. 'provenance' must be one of {"executed", "reimplementation", "cached", "unavailable"}.
    3. If 'provenance' == 'unavailable', NO numeric fields are permitted in the record.
    """
    if "provenance" not in record or not record.get("provenance"):
        raise ValueError(
            f"Schema validation failed for {source_name}: missing mandatory 'provenance' field. "
            "Rule Zero strictly forbids records without explicit provenance."
        )

    prov = str(record["provenance"]).strip().lower()
    if prov not in ALLOWED_PROVENANCE_VALUES:
        raise ValueError(
            f"Invalid provenance value '{prov}' in {source_name}. "
            f"Must be one of {sorted(list(ALLOWED_PROVENANCE_VALUES))}."
        )

    if prov == "unavailable":
        # Check that no numeric fields exist
        numeric_keys = [
            k for k, v in record.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if numeric_keys:
            raise ValueError(
                f"Rule Zero Invariant Violation in {source_name}: record marked as 'provenance: unavailable' "
                f"contains numeric fields {numeric_keys}. Unavailable records must contain NO numeric values."
            )

    return prov


@dataclass
class CellMetrics:
    model: str
    dataset: str
    run_mode: str
    seed: int
    provenance: str

    # clean/adversarial harm
    harm_clean_no_cert: Optional[float] = None
    harm_clean_shield: Optional[float] = None
    harm_clean_agentrr: Optional[float] = None
    harm_clean_verimap: Optional[float] = None
    harm_clean_pcg: Optional[float] = None

    harm_adv_no_cert: Optional[float] = None
    harm_adv_shield: Optional[float] = None
    harm_adv_agentrr: Optional[float] = None
    harm_adv_verimap: Optional[float] = None
    harm_adv_pcg: Optional[float] = None

    # R1 audit channels
    int_fail_clean: Optional[float] = None
    replay_fail_clean: Optional[float] = None
    drift_fail_clean: Optional[float] = None
    checker_fail_clean: Optional[float] = None
    covgap_fail_clean: Optional[float] = None

    utility: Optional[float] = None
    control_gain: Optional[float] = None

    token_no_cert: Optional[float] = None
    token_shield: Optional[float] = None
    token_pcg: Optional[float] = None
    latency_shield: Optional[float] = None
    latency_pcg: Optional[float] = None

    def __post_init__(self) -> None:
        validate_record_provenance(asdict(self), source_name=f"{self.model}/{self.dataset}")

    def cell_name(self) -> str:
        return f"{self.model} / {self.dataset}"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                validate_record_provenance(row, source_name=f"{path.name}:L{i}")
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            validate_record_provenance(row, source_name=f"write_jsonl:{path.name}:L{i}")
            f.write(json.dumps(row, sort_keys=True) + "\n")


def manifest_hash(rows: List[Dict[str, Any]]) -> str:
    payload = "\n".join(json.dumps(r, sort_keys=True) for r in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def assert_paper_ready(rows: List[Dict[str, Any]]) -> None:
    for i, r in enumerate(rows):
        prov = validate_record_provenance(r, source_name=f"paper_row_{i}")
        if prov not in {"executed", "reimplementation", "cached"}:
            raise RuntimeError(
                f"Paper artifact build refused: cell {r.get('model')}/{r.get('dataset')} "
                f"has unrenderable provenance '{prov}'."
            )
