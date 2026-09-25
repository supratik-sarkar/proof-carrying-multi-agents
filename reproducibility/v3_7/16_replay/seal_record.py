"""Seal the completed standalone v3.7 public record."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
CONTROL = {"RECORD_MANIFEST.json", "SHA256SUMS.txt", "RECORD_SEAL.json"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for name in CONTROL:
        (BASE / name).unlink(missing_ok=True)
    files = {}
    section_counts = Counter()
    total_bytes = 0
    for path in sorted(BASE.rglob("*")):
        if not path.is_file() or path.name == ".DS_Store" or "__pycache__" in path.parts:
            continue
        if path.is_symlink():
            raise SystemExit(f"Symlink is not permitted: {path}")
        relative = path.relative_to(BASE).as_posix()
        files[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        section_counts[relative.split("/", 1)[0]] += 1
        total_bytes += path.stat().st_size
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema": "pcg_mas_v3_7_public_record_manifest_v1",
        "project": "PCG-MAS", "scientific_release": "v3.7",
        "record_type": "PUBLIC_REPRODUCIBILITY_RECORD",
        "record_status": "SEALED_STANDALONE_RECORD",
        "sealed_at_utc": now,
        "scientific_result_authority": "14_artifacts/manuscript_stack/data",
        "total_files": len(files), "total_bytes": total_bytes,
        "section_file_counts": dict(sorted(section_counts.items())),
        "files": files,
    }
    manifest_path = BASE / "RECORD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checksums = {**{name: record["sha256"] for name, record in files.items()},
                 "RECORD_MANIFEST.json": sha256(manifest_path)}
    checksum_path = BASE / "SHA256SUMS.txt"
    checksum_path.write_text("".join(f"{value}  {name}\n" for name, value in sorted(checksums.items())))
    donor = json.loads((BASE / "15_change_control/v3_6_donor_binding.json").read_text())
    seal = {
        "schema": "pcg_mas_v3_7_record_seal_v1",
        "seal_scope": "PCG_MAS_V3_7_PUBLIC_REPRODUCIBILITY_RECORD",
        "record_created_utc": now,
        "record_manifest_sha256": sha256(manifest_path),
        "sha256sums_sha256": sha256(checksum_path),
        "donor_record_manifest_sha256": donor["donor_record_manifest_sha256"],
        "donor_record_seal_sha256": donor["donor_record_seal_sha256"],
        "status": "SEALED_STANDALONE_RECORD",
    }
    (BASE / "RECORD_SEAL.json").write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "SEALED", "files": len(files), "bytes": total_bytes}, indent=2))


if __name__ == "__main__":
    main()
