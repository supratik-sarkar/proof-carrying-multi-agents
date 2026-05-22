# scripts/v5_schema.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any
import json
import hashlib


AUDIT_CHANNELS = ["Integrity", "Replay", "Drift", "Checker", "Coverage"]
SYSTEMS = [
    "No certificate",
    "ShieldAgent",
    "PCG-MAS",
    "PCG-MAS:NoReplay",
    "PCG-MAS:NoRedundancy",
    "PCG-MAS:NoResp",
    "PCG-MAS:NoRiskCtrl",
]


@dataclass
class CellMetrics:
    model: str
    dataset: str
    run_mode: str
    seed: int

    # clean/adversarial harm
    harm_clean_no_cert: float
    harm_clean_shield: float
    harm_clean_pcg: float
    harm_adv_no_cert: float
    harm_adv_shield: float
    harm_adv_pcg: float

    # R1 audit channels
    int_fail_clean: float
    replay_fail_clean: float
    drift_fail_clean: float
    checker_fail_clean: float
    covgap_fail_clean: float

    int_fail_fresh: float
    replay_fail_fresh: float
    drift_fail_fresh: float
    checker_fail_fresh: float
    covgap_fail_fresh: float

    snapshot_replay_pass: float
    fresh_replay_match: float

    # R3
    resp_top1_closed: float
    resp_top2_open: float
    resp_multilabel_f1: float
    resp_unknown_acc: float

    # R4
    utility: float
    control_gain: float

    # R5
    token_no_cert: float
    token_shield: float
    token_pcg: float
    latency_shield: float
    latency_pcg: float

    # ablations
    harm_pcg_no_replay: float
    harm_pcg_no_redundancy: float
    harm_pcg_no_resp: float
    harm_pcg_no_riskctrl: float

    def cell_name(self) -> str:
        return f"{self.model} / {self.dataset}"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def manifest_hash(rows: List[Dict[str, Any]]) -> str:
    payload = "\n".join(json.dumps(r, sort_keys=True) for r in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def assert_paper_ready(rows: List[Dict[str, Any]]) -> None:
    bad = [r for r in rows if r.get("run_mode") != "full"]
    if bad:
        example = bad[0]
        raise RuntimeError(
            "Paper artifact build refused: non-full run_mode detected. "
            f"Example cell={example.get('model')} / {example.get('dataset')}, "
            f"run_mode={example.get('run_mode')!r}."
        )
    for r in rows:
        mode = r.get("shieldagent_implementation_mode")
        if mode != "official_authors_pipeline":
            raise RuntimeError(
                f"Paper build requires official ShieldAgent mode; got {mode} "
                f"for {(r.get('model'), r.get('dataset'), r.get('seed'))}"
            )


def drift_fail(snapshot_replay_pass: bool, fresh_replay_match: bool) -> int:
    return int(bool(snapshot_replay_pass) and not bool(fresh_replay_match))