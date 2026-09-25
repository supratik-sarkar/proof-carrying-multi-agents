"""PCG-MAS v3.4R Comprehensive Q1 Code & Configuration Freeze.

Hashes EVERY artifact capable of changing acceptance or Q2 semantics:
- all src/pcg/v3_4r/*.py modules
- scripts/v3_4r/run_v3_4r_offline.py
- PCG_MAS_V3_4R_OFFLINE_REMEDIATION_PACKAGE/RUN_V3_4R_OFFLINE.sh
- All v3.4R package contract JSON / MD files
- Parent v3.4 authority manifest hash
- Runtime-only inputs SHA-256 (if generated)

Produces: V3_4R_Q1_FREEZE_SHA256
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_q1_freeze_manifest(
    repo_root: Path,
    pkg_root: Path,
    runtime_inputs_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], str]:
    """Scans and hashes all acceptance-affecting sources and configurations.

    Returns:
        (manifest_dict, overall_freeze_sha256)
    """
    file_hashes: Dict[str, str] = {}

    # 1. All python modules in src/pcg/v3_4r/
    src_dir = repo_root / "src" / "pcg" / "v3_4r"
    for py_path in sorted(src_dir.glob("*.py")):
        rel_path = str(py_path.relative_to(repo_root))
        file_hashes[rel_path] = compute_file_sha256(py_path)

    # 2. scripts/v3_4r/run_v3_4r_offline.py
    runner_py = repo_root / "scripts" / "v3_4r" / "run_v3_4r_offline.py"
    if runner_py.exists():
        file_hashes[str(runner_py.relative_to(repo_root))] = (
            compute_file_sha256(runner_py)
        )

    # 3. RUN_V3_4R_OFFLINE.sh in pkg_root
    run_sh = pkg_root / "RUN_V3_4R_OFFLINE.sh"
    if run_sh.exists():
        file_hashes[str(run_sh.relative_to(repo_root))] = compute_file_sha256(
            run_sh
        )

    # 4. Package contracts and specifications
    for json_path in sorted(pkg_root.glob("*.json")):
        rel_path = str(json_path.relative_to(repo_root))
        file_hashes[rel_path] = compute_file_sha256(json_path)

    for md_path in sorted(pkg_root.glob("*.md")):
        rel_path = str(md_path.relative_to(repo_root))
        file_hashes[rel_path] = compute_file_sha256(md_path)

    # 5. Parent v3.4 authority manifest
    parent_manifest = (
        repo_root
        / "PCG_MAS_V3_4_EXPERIMENTAL_FREEZE_PACKAGE"
        / "PACKAGE_MANIFEST_SHA256.json"
    )
    if parent_manifest.exists():
        file_hashes[str(parent_manifest.relative_to(repo_root))] = (
            compute_file_sha256(parent_manifest)
        )

    # 6. Frozen calibration parameter file from parent v3.4
    cal_file = (
        repo_root
        / "artifacts"
        / "v3_4"
        / "experimental_controller"
        / "V34-G4"
        / "A15_V3_4_FROZEN_CALIBRATION.json"
    )
    if cal_file.exists():
        file_hashes[str(cal_file.relative_to(repo_root))] = (
            compute_file_sha256(cal_file)
        )

    # 7. Runtime-only inputs hash if generated
    if runtime_inputs_path and runtime_inputs_path.exists():
        rel_inp = str(runtime_inputs_path.relative_to(repo_root))
        file_hashes[rel_inp] = compute_file_sha256(runtime_inputs_path)

    manifest = {
        "schema": "PCG_MAS_V3_4R_Q1_FREEZE_MANIFEST_V1",
        "total_files_frozen": len(file_hashes),
        "files": file_hashes,
    }

    serialized = json.dumps(manifest, sort_keys=True, indent=2)
    overall_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    return manifest, overall_sha256


def verify_q1_freeze(
    manifest: Dict[str, Any], expected_sha256: str, repo_root: Path
) -> Tuple[bool, Optional[str]]:
    """Verifies that all frozen files match their recorded SHA-256 hashes."""
    for rel_path, expected_hash in manifest.get("files", {}).items():
        p = repo_root / rel_path
        if not p.exists():
            return False, f"Missing frozen file: {rel_path}"
        curr_hash = compute_file_sha256(p)
        if curr_hash != expected_hash:
            return (
                False,
                f"Hash mismatch on {rel_path}: expected {expected_hash}, got {curr_hash}",
            )

    serialized = json.dumps(manifest, sort_keys=True, indent=2)
    actual_overall = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if actual_overall != expected_sha256:
        return (
            False,
            f"Overall freeze SHA-256 mismatch: expected {expected_sha256}, got {actual_overall}",
        )

    return True, None
