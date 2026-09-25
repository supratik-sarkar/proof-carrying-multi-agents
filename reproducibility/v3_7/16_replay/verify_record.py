"""Verify the standalone v3.7 record, projections, authority, privacy, and seal."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
CONTROL = {"RECORD_MANIFEST.json", "SHA256SUMS.txt", "RECORD_SEAL.json"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(relative: str) -> Path:
    path = BASE / relative
    if path.is_symlink() or not path.resolve().is_relative_to(BASE.resolve()):
        raise AssertionError(f"Unsafe record path: {relative}")
    return path


def load_jsonl(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def verify_privacy() -> int:
    denied = [bytes.fromhex(value).decode() for value in (
        "63686174677074", "636f646578", "616e746967726176697479", "636c61756465",
        "69636c72", "6e657572697073", "686973746f726963616c5f6d6f636b",
        "5f6d6f636b", "73796e746865746963",
    )]
    private_fragments = ["/" + "Users/", "/" + "home/", "file" + "://"]
    text_suffixes = {".csv", ".json", ".jsonl", ".md", ".py", ".sh", ".tex", ".txt"}
    checked = 0
    for path in BASE.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        checked += 1
        content = path.read_text(errors="replace")
        lower = content.lower()
        name = path.relative_to(BASE).as_posix().lower()
        for term in denied:
            assert term.lower() not in lower and term.lower() not in name, path.relative_to(BASE)
        for fragment in private_fragments:
            assert fragment.lower() not in lower, path.relative_to(BASE)
        assert not re.search(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", content), path.relative_to(BASE)
        assert not re.search(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{8,}", content), path.relative_to(BASE)
        for match in re.finditer(r'(?m)^\s*"?([a-zA-Z0-9_<>.%-]+)\s+\d+\s+\d+\.\d+\s+\d+\.\d+', content):
            process_user = match.group(1).strip('"').lower()
            assert (
                process_user in {"user", "root", "daemon", "nobody", "runner", "<local_user>"}
                or process_user.startswith("<local_")
                or process_user.startswith("local_")
            ), f"Unsanitized process username {process_user} in {path.relative_to(BASE)}"
    return checked


def main() -> None:
    manifest_path = BASE / "RECORD_MANIFEST.json"
    checksum_path = BASE / "SHA256SUMS.txt"
    seal_path = BASE / "RECORD_SEAL.json"
    manifest = json.loads(manifest_path.read_text())
    seal = json.loads(seal_path.read_text())
    expected = manifest["files"]
    actual = {
        path.relative_to(BASE).as_posix() for path in BASE.rglob("*")
        if path.is_file() and path.name != ".DS_Store" and "__pycache__" not in path.parts
    }
    assert actual == set(expected) | CONTROL, "Record membership mismatch"
    for name, record in expected.items():
        path = safe_path(name)
        assert path.is_file(), name
        assert path.stat().st_size == record["bytes"], name
        assert sha256(path) == record["sha256"], name
    checksum_rows = {}
    for line in checksum_path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        checksum_rows[name] = digest
        assert sha256(safe_path(name)) == digest, name
    assert set(checksum_rows) == set(expected) | {"RECORD_MANIFEST.json"}
    assert seal["record_manifest_sha256"] == sha256(manifest_path)
    assert seal["sha256sums_sha256"] == sha256(checksum_path)

    donor = json.loads((BASE / "15_change_control/v3_6_donor_binding.json").read_text())
    assert donor["donor_record_manifest_sha256"] == seal["donor_record_manifest_sha256"]
    assert donor["donor_record_seal_sha256"] == seal["donor_record_seal_sha256"]
    admitted = 0
    for row in load_jsonl(BASE / "15_change_control/donor_artifact_binding.jsonl"):
        admitted += 1
        path = safe_path(row["public_path"])
        assert sha256(path) == row["public_projection_sha256"], row["identity"]
        assert row["identity_hash_reconciliation"] == "PASS"
    assert admitted == donor["admitted_artifacts"]

    raw = list(load_jsonl(BASE / "05_generations/raw_response_manifest.jsonl"))
    assert len(raw) == 2240
    for row in raw:
        assert sha256(safe_path(row["public_rel_path"])) == row["public_projection_sha256"]
        assert row["identity_reconciliation"] == "PASS"
    traces = list(load_jsonl(BASE / "06_agent_traces/trajectory_manifest.jsonl"))
    assert len(traces) == 434
    for row in traces:
        assert sha256(safe_path(row["public_rel_path"])) == row["public_projection_sha256"]
    with (BASE / "10_execution/run_registry.csv").open(newline="") as handle:
        receipts = list(csv.DictReader(handle))
    assert len(receipts) == 140
    for row in receipts:
        assert sha256(safe_path(row["public_receipt_path"])) == row["public_projection_sha256"]

    with (BASE / "07_verification/verifier_scores.csv").open(newline="") as handle:
        scores = list(csv.DictReader(handle))
    assert scores and all(row["value_status"] == "NOT_RETAINED" and not row["score"] for row in scores)
    for name, field in (("acceptance_decisions.jsonl", "decision_status"),
                        ("certificate_ledger.jsonl", "certificate_status"),
                        ("obligation_scores.jsonl", "obligation_status"),
                        ("native_contract_results.jsonl", "contract_status")):
        rows = list(load_jsonl(BASE / "08_certificates" / name))
        assert len(rows) == 2240 and all(row[field] == "NOT_RETAINED" for row in rows), name
    authority = BASE / manifest["scientific_result_authority"]
    assert authority == BASE / "14_artifacts/manuscript_stack/data" and authority.is_dir()
    assert sum(1 for _ in csv.DictReader((authority / "cell_metrics.csv").open())) == 392
    privacy_files = verify_privacy()
    print(json.dumps({
        "status": "PASS", "files": len(expected), "admitted_artifacts": admitted,
        "raw_responses": len(raw), "raw_trajectories": len(traces),
        "execution_receipts": len(receipts), "privacy_files_scanned": privacy_files,
        "external_private_tree_dependency": False,
    }, indent=2))


if __name__ == "__main__":
    main()
