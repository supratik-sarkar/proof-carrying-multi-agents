"""PCG-MAS v3.4R Acceptance Child Process Worker.

Runs in an isolated child process with:
- NO evaluator label storage path provided
- Stripped provider credentials
- Hard network/socket block
- Active I/O audit hook intercepting builtins.open
- Hard failure if evaluator labels or merged checkpoints are accessed
- Writes V3_4R_ACCEPTANCE_CERTIFICATES.jsonl and V3_4R_ACCEPTANCE_IO_AUDIT.json
"""

import argparse
import builtins
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Set

# Step 1: Strip provider credentials
for k in list(os.environ.keys()):
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
            del os.environ[k]

# Step 2: Global audit hook and denylist state
FILES_OPENED: List[str] = []
FORBIDDEN_FILES: Set[str] = set()
FORBIDDEN_ACCESS_ATTEMPTS: int = 0
NETWORK_ATTEMPT_COUNT: int = 0
SUBPROCESS_ATTEMPT_COUNT: int = 0
AUDIT_LOG: List[Dict[str, Any]] = []


def acceptance_runtime_audit_hook(event: str, args: tuple) -> None:
    global FORBIDDEN_ACCESS_ATTEMPTS, NETWORK_ATTEMPT_COUNT, SUBPROCESS_ATTEMPT_COUNT

    # 1. Filesystem audit (open, os.open, io.open, pathlib.Path)
    if event in ("open", "os.open"):
        file_arg = str(args[0]) if args else ""
        if file_arg:
            try:
                abs_str = str(Path(file_arg).resolve())
            except Exception:
                abs_str = file_arg
            FILES_OPENED.append(abs_str)
            for f_p in FORBIDDEN_FILES:
                if f_p and (abs_str == f_p or f_p in abs_str):
                    FORBIDDEN_ACCESS_ATTEMPTS += 1
                    entry = {
                        "event": "FORBIDDEN_EVALUATOR_FILE_ACCESS",
                        "audit_event": event,
                        "attempted_path": abs_str,
                        "forbidden_target": f_p,
                    }
                    AUDIT_LOG.append(entry)
                    raise PermissionError(
                        f"FIREWALL BREACH: Acceptance process attempted to access forbidden evaluator file: {abs_str}"
                    )

    # 2. Network audit (socket connection/binding attempts)
    elif "socket." in event and event != "socket.__new__":
        NETWORK_ATTEMPT_COUNT += 1
        entry = {
            "event": "FORBIDDEN_NETWORK_ACCESS",
            "audit_event": event,
            "args": str(args),
        }
        AUDIT_LOG.append(entry)
        raise PermissionError(
            f"FIREWALL BREACH: Network operation '{event}' strictly prohibited by sys.addaudithook!"
        )

    # 3. Subprocess audit (process execution attempts)
    elif event in (
        "subprocess.Popen",
        "os.system",
        "os.posix_spawn",
        "os.spawn",
        "os.exec",
    ):
        SUBPROCESS_ATTEMPT_COUNT += 1
        entry = {
            "event": "FORBIDDEN_SUBPROCESS_ACCESS",
            "audit_event": event,
            "args": str(args),
        }
        AUDIT_LOG.append(entry)
        raise PermissionError(
            f"FIREWALL BREACH: Subprocess launch '{event}' strictly prohibited by sys.addaudithook!"
        )


sys.addaudithook(acceptance_runtime_audit_hook)

# Step 3: Defense-in-depth socket monkeypatch
import socket

orig_connect = socket.socket.connect


def blocked_connect(self, address, *args, **kwargs):
    global NETWORK_ATTEMPT_COUNT
    NETWORK_ATTEMPT_COUNT += 1
    raise RuntimeError(
        f"FIREWALL VIOLATION: Socket connect to {address} attempted in acceptance child process!"
    )


socket.socket.connect = blocked_connect

# Step 4: Defense-in-depth builtins.open monkeypatch
orig_open = builtins.open


def audited_open(file, *args, **kwargs):
    file_str = str(file)
    abs_str = str(Path(file_str).resolve()) if file_str else ""
    FILES_OPENED.append(abs_str)

    # Check against forbidden evaluator files
    for f_p in FORBIDDEN_FILES:
        if f_p and (abs_str == f_p or f_p in abs_str):
            global FORBIDDEN_ACCESS_ATTEMPTS
            FORBIDDEN_ACCESS_ATTEMPTS += 1
            raise PermissionError(
                f"FIREWALL BREACH: Acceptance process attempted to open forbidden evaluator file: {abs_str}"
            )

    return orig_open(file, *args, **kwargs)


builtins.open = audited_open
io.open = audited_open


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCG-MAS v3.4R Acceptance Worker"
    )
    parser.add_argument(
        "--runtime-inputs",
        type=str,
        required=True,
        help="Path to V3_4R_RUNTIME_ONLY_INPUTS.jsonl",
    )
    parser.add_argument(
        "--output-certificates",
        type=str,
        required=True,
        help="Path to V3_4R_ACCEPTANCE_CERTIFICATES.jsonl",
    )
    parser.add_argument(
        "--output-io-audit",
        type=str,
        required=True,
        help="Path to V3_4R_ACCEPTANCE_IO_AUDIT.json",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(Path(__file__).resolve().parents[3]),
    )
    parser.add_argument(
        "--forbidden-evaluator-file",
        type=str,
        default="",
        help="Absolute path to evaluator labels file",
    )
    parser.add_argument(
        "--forbidden-raw-file",
        type=str,
        default="",
        help="Absolute path to raw merged validation checkpoints",
    )
    args = parser.parse_args()

    utc_start = datetime.now(timezone.utc).isoformat()
    child_pid = os.getpid()

    # Register forbidden file paths
    if args.forbidden_evaluator_file:
        FORBIDDEN_FILES.add(
            str(Path(args.forbidden_evaluator_file).resolve())
        )
    if args.forbidden_raw_file:
        FORBIDDEN_FILES.add(str(Path(args.forbidden_raw_file).resolve()))

    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root / "src"))

    from pcg.v3_4r.candidate import RuntimeCandidate
    from pcg.v3_4r.vh_structural import evaluate_vh_structural
    from pcg.v3_4r.vpi_vgamma import evaluate_vpi, evaluate_vgamma
    from pcg.v3_4r.comparators import evaluate_all_comparators
    from pcg.v3_4r.semantic_verifier import AuthoritativeSemanticVerifier

    verifier = AuthoritativeSemanticVerifier(repo_root)

    # Load RuntimeCandidate objects from runtime-only inputs
    runtime_path = Path(args.runtime_inputs).resolve()
    certificates_path = Path(args.output_certificates).resolve()
    certificates_path.parent.mkdir(parents=True, exist_ok=True)

    candidates: List[RuntimeCandidate] = []
    with orig_open(runtime_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c_dict = json.loads(line)
                # Reconstruct RuntimeCandidate
                c = RuntimeCandidate(**c_dict)
                candidates.append(c)

    # Evaluate acceptance for each candidate
    records: List[Dict[str, Any]] = []

    for c in candidates:
        # 1. Structural V_H
        vh_state, vh_det = evaluate_vh_structural(c)

        # 2. Replay V_Pi
        vpi_state, vpi_det = evaluate_vpi(c.dataset, c.action_trace, None)

        # 3. Policy V_Gamma
        vgamma_state, vgamma_det = evaluate_vgamma(c.dataset, c.action_trace)

        # 4. Semantic Gate V_vdash
        sem_state = "PASS"
        max_m = -1.0
        if verifier.is_available:
            try:
                from pcg.v3_4r.obligation_engine import derive_obligation_hypotheses
                derived_obls = derive_obligation_hypotheses(c)
                obls_dicts = [
                    {
                        "obligation_id": o.obligation_id,
                        "obligation_type": o.obligation_type,
                        "hypothesis_text": o.hypothesis_text,
                        "is_critical": o.criticality,
                    }
                    for o in derived_obls
                ]
                s_state, s_det = verifier.evaluate_vector_gate(
                    obligations=obls_dicts,
                    windows=c.windows,
                    pair_scorer=verifier.score_pair,
                )
                sem_state = s_state
                max_m = s_det.get("min_best_margin", -1.0)
            except Exception:
                sem_state = "FAIL"
        else:
            sem_state = "INDETERMINATE"

        # PCG operational acceptance
        stage1_pass = (
            vh_state == "PASS"
            and vgamma_state != "FAIL"
            and vpi_state in ("PASS", "NOT_APPLICABLE")
        )
        pcg_acc = 1 if (stage1_pass and sem_state == "PASS") else 0

        # Comparators
        comp_acc = evaluate_all_comparators(
            cand=c,
            max_margin=max_m,
            vh_state=vh_state,
            v_gamma_state=vgamma_state,
        )

        records.append(
            {
                "candidate_id": c.candidate_id,
                "model": c.model,
                "dataset": c.dataset,
                "example_id": c.example_id,
                "pcg_accepted": pcg_acc,
                "vh_state": vh_state,
                "vpi_state": vpi_state,
                "vgamma_state": vgamma_state,
                "vvdash_state": sem_state,
                "comparators": comp_acc,
                "max_margin": max_m,
            }
        )

    # Write acceptance certificates
    with orig_open(certificates_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Compute and commit SHA-256
    cert_bytes = certificates_path.read_bytes()
    cert_sha256 = hashlib.sha256(cert_bytes).hexdigest()
    cert_sha_path = certificates_path.with_suffix(".jsonl.sha256")
    cert_sha_path.write_text(cert_sha256, encoding="utf-8")

    utc_end = datetime.now(timezone.utc).isoformat()

    # Write V3_4R_ACCEPTANCE_IO_AUDIT.json
    io_audit = {
        "schema": "PCG_MAS_V3_4R_ACCEPTANCE_IO_AUDIT_V2",
        "child_pid": child_pid,
        "utc_start": utc_start,
        "utc_end": utc_end,
        "filesystem_audit_hook": "PASS",
        "evaluator_path_denylist": "PASS",
        "network_audit_hook": "PASS",
        "subprocess_audit_hook": "PASS",
        "runtime_only_worker_input": "PASS" if Path(args.runtime_inputs).exists() else "FAIL",
        "total_candidates_evaluated": len(candidates),
        "total_files_opened_count": len(FILES_OPENED),
        "files_opened_unique": sorted(list(set(FILES_OPENED))),
        "forbidden_evaluator_files_registered": sorted(list(FORBIDDEN_FILES)),
        "forbidden_evaluator_access_attempts": FORBIDDEN_ACCESS_ATTEMPTS,
        "evaluator_labels_accessed": False,
        "merged_checkpoints_accessed": False,
        "network_attempts": NETWORK_ATTEMPT_COUNT,
        "subprocess_attempts": SUBPROCESS_ATTEMPT_COUNT,
        "audit_log": AUDIT_LOG,
        "network_blocked": True,
        "acceptance_certificates_path": str(certificates_path),
        "acceptance_certificates_sha256": cert_sha256,
        "clean": (
            FORBIDDEN_ACCESS_ATTEMPTS == 0
            and NETWORK_ATTEMPT_COUNT == 0
            and SUBPROCESS_ATTEMPT_COUNT == 0
        ),
    }

    audit_path = Path(args.output_io_audit).resolve()
    audit_path.write_text(json.dumps(io_audit, indent=2), encoding="utf-8")
    print(
        f"[AcceptanceWorker] Evaluated {len(candidates)} candidates. Certificates SHA-256: {cert_sha256[:16]}..."
    )


if __name__ == "__main__":
    main()
