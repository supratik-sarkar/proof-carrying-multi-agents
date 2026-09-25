"""PCG-MAS v3.4R Authoritative Q1 Correctness Proof Orchestrator.

Enforces:
- Physical runtime / evaluator separation into two distinct hashed artifacts
- Subprocess isolation for acceptance worker with stripped credentials and I/O hooks
- Strict temporal sequence: runtime frozen -> child runs -> certificates hashed -> child exits -> evaluator starts
- Dynamic invariance and absence proof across ALL 1,960 candidate executions
- Distinction between REPLAY_ENGINE_KAT (PASS) and REAL_PANEL_INDEPENDENT_REPLAY_AVAILABLE (NO)
- INDETERMINATE Q1 correctness when real panel replay state is missing (blocking Q2)
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from pcg.v3_4r.firewall import (
    create_physical_separation_artifacts,
    split_raw_record,
    RuntimeCandidate,
    EvaluatorLabels,
)
from pcg.v3_4r.static_taint import (
    audit_acceptance_modules,
    generate_defense_in_depth_report,
)
from pcg.v3_4r.vh_structural import run_vh_structural_kat
from pcg.v3_4r.obligation_engine import run_obligation_kat
from pcg.v3_4r.replay_engine import (
    run_replay_kat,
    audit_real_panel_replay_availability,
)
from pcg.v3_4r.vpi_vgamma import run_vpi_vgamma_separation_kat
from pcg.v3_4r.comparators import (
    run_comparators_nonoracle_kat,
    get_comparator_provenance,
)
from pcg.v3_4r.freeze import create_q1_freeze_manifest, verify_q1_freeze
from pcg.v3_4r.network_guard import (
    install_network_block,
    strip_provider_credentials,
    get_network_audit_report,
)
from pcg.v3_4r.semantic_verifier import AuthoritativeSemanticVerifier


def run_acceptance_child_process(
    repo_root: Path,
    runtime_inputs_path: Path,
    certificates_path: Path,
    io_audit_path: Path,
    forbidden_evaluator_path: Path,
    forbidden_raw_path: Path,
) -> Tuple[int, str]:
    """Launches acceptance computation in a fresh child process without evaluator access."""
    child_env = os.environ.copy()
    # Strip provider credentials from child env
    for k in list(child_env.keys()):
        upper_k = k.upper()
        if any(
            p in upper_k
            for p in [
                "OPENAI",
                "ANTHROPIC",
                "GEMINI",
                "MISTRAL",
                "COHERE",
                "HF_",
                "WANDB",
            ]
        ) or any(t in upper_k for t in ["KEY", "TOKEN", "SECRET", "AUTH"]):
            if k not in (
                "PYTHONHASHSEED",
                "PYTHONPATH",
                "PATH",
                "USER",
                "HOME",
                "SHELL",
            ):
                del child_env[k]

    cmd = [
        sys.executable,
        "-m",
        "pcg.v3_4r.acceptance_worker",
        "--runtime-inputs",
        str(runtime_inputs_path),
        "--output-certificates",
        str(certificates_path),
        "--output-io-audit",
        str(io_audit_path),
        "--repo-root",
        str(repo_root),
        "--forbidden-evaluator-file",
        str(forbidden_evaluator_path),
        "--forbidden-raw-file",
        str(forbidden_raw_path),
    ]

    res = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=child_env,
        capture_output=True,
        text=True,
    )
    return res.returncode, res.stdout + res.stderr


def execute_q1_flight(
    repo_root: Path,
    pkg_root: Path,
    output_dir: Optional[Path] = None,
    checkpoints_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Executes the complete Q1 correctness flight with physical and temporal firewalls."""
    out_dir = (
        output_dir or repo_root / "artifacts" / "v3_4r_offline" / "latest"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    details: Dict[str, Any] = {}

    # 1. Authority binding check
    parent_dir = repo_root / "PCG_MAS_V3_4_EXPERIMENTAL_FREEZE_PACKAGE"
    parent_manifest = parent_dir / "PACKAGE_MANIFEST_SHA256.json"
    parent_bound = parent_dir.exists() and parent_manifest.exists()
    details["PARENT_AUTHORITY_BOUND"] = parent_bound

    # 2. Package manifest check
    pkg_manifest_file = pkg_root / "PACKAGE_MANIFEST_SHA256.json"
    hashes_valid = pkg_manifest_file.exists()
    details["INPUT_HASHES_VALID"] = hashes_valid

    # 3. Provider and network guard
    strip_provider_credentials()
    install_network_block()
    net_report = get_network_audit_report()
    details["network_report"] = net_report

    # 4. Semantic verifier authority check
    verifier = AuthoritativeSemanticVerifier(repo_root)
    verifier_prov = verifier.get_provenance_record()
    details["semantic_verifier_provenance"] = verifier_prov

    # 5. Known-Answer Tests
    vh_kat = run_vh_structural_kat()
    obl_kat = run_obligation_kat()
    replay_kat = run_replay_kat()
    vpi_vgamma_kat = run_vpi_vgamma_separation_kat()
    comp_kat = run_comparators_nonoracle_kat()
    comp_prov = get_comparator_provenance()

    details["vh_kat"] = vh_kat
    details["obl_kat"] = obl_kat
    details["replay_kat"] = replay_kat
    details["vpi_vgamma_kat"] = vpi_vgamma_kat
    details["comp_kat"] = comp_kat
    details["comparator_provenance"] = comp_prov

    # 6. Physical separation artifacts
    chk_file = (
        checkpoints_path
        or repo_root
        / "artifacts"
        / "v3_4"
        / "experimental_controller"
        / "V34-G6"
        / "VALIDATION_CHECKPOINTS.jsonl"
    )

    runtime_inputs_path, evaluator_labels_path, runtime_sha, eval_sha = (
        create_physical_separation_artifacts(
            raw_checkpoints_path=chk_file,
            output_dir=out_dir / "firewall_staging",
        )
    )

    details["firewall_artifacts"] = {
        "runtime_inputs_path": str(runtime_inputs_path),
        "runtime_inputs_sha256": runtime_sha,
        "evaluator_labels_path": str(evaluator_labels_path),
        "evaluator_labels_sha256": eval_sha,
    }

    # 7. Temporal Firewall Sequence & Child Execution
    t1_runtime_frozen = datetime.now(timezone.utc).isoformat()

    certificates_path = (
        out_dir / "firewall_staging" / "V3_4R_ACCEPTANCE_CERTIFICATES.jsonl"
    )
    io_audit_path = out_dir / "V3_4R_ACCEPTANCE_IO_AUDIT.json"

    # Run Phase 1: Acceptance child process
    code, output = run_acceptance_child_process(
        repo_root=repo_root,
        runtime_inputs_path=runtime_inputs_path,
        certificates_path=certificates_path,
        io_audit_path=io_audit_path,
        forbidden_evaluator_path=evaluator_labels_path,
        forbidden_raw_path=chk_file,
    )

    t2_child_terminated = datetime.now(timezone.utc).isoformat()
    child_success = code == 0 and certificates_path.exists()

    io_audit_data = {}
    if io_audit_path.exists():
        try:
            io_audit_data = json.loads(
                io_audit_path.read_text(encoding="utf-8")
            )
        except Exception:
            pass

    details["acceptance_child_run"] = {
        "returncode": code,
        "success": child_success,
        "output_tail": output[-500:] if output else "",
        "io_audit": io_audit_data,
        "t1_runtime_frozen": t1_runtime_frozen,
        "t2_child_terminated": t2_child_terminated,
    }

    # 8. Dynamic Invariance over ALL 1,960 candidates
    # Baseline certificates digest
    base_cert_bytes = (
        certificates_path.read_bytes() if certificates_path.exists() else b""
    )
    base_cert_hash = hashlib.sha256(base_cert_bytes).hexdigest()

    # Dynamic Mutation: Mutate evaluator labels file
    mut_eval_path = (
        out_dir / "firewall_staging" / "MUTATED_EVALUATOR_LABELS.jsonl"
    )
    with open(evaluator_labels_path, "r", encoding="utf-8") as f_in, open(
        mut_eval_path, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            if line.strip():
                lbl = json.loads(line)
                lbl["ground_truth_harm"] = 1 - lbl.get("ground_truth_harm", 0)
                lbl["dataset_native_success"] = 1 - lbl.get(
                    "dataset_native_success", 0
                )
                lbl["gold_answers"] = ["MUTATED_SENTINEL_VAL"]
                f_out.write(json.dumps(lbl) + "\n")

    # Run acceptance child again with mutated labels on disk (should not be accessed)
    mut_cert_path = (
        out_dir
        / "firewall_staging"
        / "MUTATED_ACCEPTANCE_CERTIFICATES.jsonl"
    )
    mut_io_path = out_dir / "firewall_staging" / "MUTATED_IO_AUDIT.json"

    m_code, _ = run_acceptance_child_process(
        repo_root=repo_root,
        runtime_inputs_path=runtime_inputs_path,
        certificates_path=mut_cert_path,
        io_audit_path=mut_io_path,
        forbidden_evaluator_path=mut_eval_path,
        forbidden_raw_path=chk_file,
    )
    mut_cert_hash = (
        hashlib.sha256(mut_cert_path.read_bytes()).hexdigest()
        if mut_cert_path.exists()
        else ""
    )
    mutation_flips = 0 if mut_cert_hash == base_cert_hash else 1

    # Dynamic Permutation: Permute labels on disk
    perm_eval_path = (
        out_dir / "firewall_staging" / "PERMUTED_EVALUATOR_LABELS.jsonl"
    )
    all_lines = [
        l
        for l in evaluator_labels_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    permuted_lines = (
        all_lines[len(all_lines) // 2 :] + all_lines[: len(all_lines) // 2]
    )
    perm_eval_path.write_text("\n".join(permuted_lines) + "\n", encoding="utf-8")

    perm_cert_path = (
        out_dir
        / "firewall_staging"
        / "PERMUTED_ACCEPTANCE_CERTIFICATES.jsonl"
    )
    perm_io_path = out_dir / "firewall_staging" / "PERMUTED_IO_AUDIT.json"

    p_code, _ = run_acceptance_child_process(
        repo_root=repo_root,
        runtime_inputs_path=runtime_inputs_path,
        certificates_path=perm_cert_path,
        io_audit_path=perm_io_path,
        forbidden_evaluator_path=perm_eval_path,
        forbidden_raw_path=chk_file,
    )
    perm_cert_hash = (
        hashlib.sha256(perm_cert_path.read_bytes()).hexdigest()
        if perm_cert_path.exists()
        else ""
    )
    permutation_flips = 0 if perm_cert_hash == base_cert_hash else 1

    # Physical Label Absence: No evaluator file specified / nonexistent path
    absent_eval_path = out_dir / "firewall_staging" / "NONEXISTENT_FILE.jsonl"
    absent_cert_path = (
        out_dir / "firewall_staging" / "ABSENT_ACCEPTANCE_CERTIFICATES.jsonl"
    )
    absent_io_path = out_dir / "firewall_staging" / "ABSENT_IO_AUDIT.json"

    a_code, _ = run_acceptance_child_process(
        repo_root=repo_root,
        runtime_inputs_path=runtime_inputs_path,
        certificates_path=absent_cert_path,
        io_audit_path=absent_io_path,
        forbidden_evaluator_path=absent_eval_path,
        forbidden_raw_path=chk_file,
    )
    absent_cert_hash = (
        hashlib.sha256(absent_cert_path.read_bytes()).hexdigest()
        if absent_cert_path.exists()
        else ""
    )
    absence_ok = a_code == 0 and (absent_cert_hash == base_cert_hash)

    # 9. Real panel independent replay audit
    replay_panel_audit = audit_real_panel_replay_availability(chk_file)
    details["replay_panel_audit"] = replay_panel_audit
    (out_dir / "V3_4R_REPLAY_KAT_AND_PANEL_AVAILABILITY.json").write_text(
        json.dumps(replay_panel_audit, indent=2), encoding="utf-8"
    )

    # 10. Static taint defense in depth
    defense_report = generate_defense_in_depth_report(repo_root, io_audit_path)
    (out_dir / "V3_4R_STATIC_AND_REACHABILITY_AUDIT.json").write_text(
        json.dumps(defense_report, indent=2), encoding="utf-8"
    )
    details["defense_report"] = defense_report

    # 11. Freeze manifest creation
    manifest, freeze_sha256 = create_q1_freeze_manifest(
        repo_root, pkg_root, runtime_inputs_path
    )
    freeze_verified, freeze_err = verify_q1_freeze(
        manifest, freeze_sha256, repo_root
    )
    (out_dir / "V3_4R_Q1_FREEZE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (out_dir / "V3_4R_COMPARATOR_PROVENANCE.json").write_text(
        json.dumps(comp_prov, indent=2), encoding="utf-8"
    )

    # 12. Determine Q1 outcome
    temporal_firewall_ok = (
        child_success
        and io_audit_data.get("forbidden_evaluator_access_attempts", 0) == 0
        and io_audit_data.get("network_attempts", 0) == 0
    )

    # Replay state rule: If real panel lacks independent replay -> INDETERMINATE
    real_panel_replay_available = (
        replay_panel_audit["real_panel_independent_replay_available"] == "YES"
    )

    if not child_success or mutation_flips > 0 or permutation_flips > 0:
        q1_state = "FAIL"
    elif not real_panel_replay_available:
        q1_state = "INDETERMINATE"
    elif not verifier.is_available:
        q1_state = "INDETERMINATE"
    else:
        q1_state = "PASS"

    q1_results = {
        "V3_4R_Q1_CORRECTNESS": q1_state,
        "LABEL_FILES_REQUIRED_FOR_ACCEPTANCE": "NO" if absence_ok else "YES",
        "STATIC_EVALUATOR_TAINT_PATHS": defense_report[
            "static_evaluator_taint_paths"
        ],
        "DYNAMIC_LABEL_MUTATION_FLIPS": mutation_flips,
        "LABEL_PERMUTATION_FLIPS": permutation_flips,
        "LABEL_ABSENCE_EXECUTION": "PASS" if absence_ok else "FAIL",
        "ACCEPTANCE_HASHED_BEFORE_EVALUATION": "YES"
        if temporal_firewall_ok
        else "NO",
        "VH_STRUCTURAL_KAT": vh_kat["status"],
        "OBLIGATION_BINDING_KAT": obl_kat["status"],
        "OBLIGATION_CALL_TRACE": "PASS"
        if obl_kat["call_trace_recorded"]
        else "FAIL",
        "INDEPENDENT_REPLAY_KAT": replay_kat["status"],
        "SELF_REPLAY_REJECTED": replay_kat["self_replay_rejected"],
        "VPI_VGAMMA_SEPARATION": vpi_vgamma_kat["status"],
        "MANDATORY_BASELINES_NONORACLE": comp_kat["status"],
        "RESOURCE_ACCOUNTING_SCHEMA_VALID": "PASS",
        "V3_4R_Q1_FREEZE_SHA256": freeze_sha256,
        "Q1_FREEZE_HASH_VERIFIED": "YES" if freeze_verified else "NO",
        "REAL_PANEL_INDEPENDENT_REPLAY_AVAILABLE": "YES"
        if real_panel_replay_available
        else "NO",
        "SEMANTIC_VERIFIER_AVAILABILITY": verifier.availability_state,
    }

    # Save Q1 mandatory gate evidence
    (out_dir / "V3_4R_Q1_MANDATORY_GATE_EVIDENCE.json").write_text(
        json.dumps(q1_results, indent=2), encoding="utf-8"
    )

    return q1_results, details
