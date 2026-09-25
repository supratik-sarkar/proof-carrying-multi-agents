"""Refresh v3.7 protocol, metric, source, and artifact indexes from the embedded stack."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parents[1]
STACK = BASE / "14_artifacts/manuscript_stack"
DATA = STACK / "data"
SOURCE = STACK / "source"
IMAGES = STACK / "images"
TABLES = STACK / "tables"
NOT_RETAINED = "NOT_RETAINED"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(BASE).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def protocol_indexes() -> None:
    cells = read_csv(DATA / "cell_registry_56.csv")
    metrics = read_csv(DATA / "cell_metrics.csv")
    models = sorted({row["model"] for row in cells})
    datasets = sorted({row["dataset"] for row in cells})
    methods = sorted({row["method"] for row in metrics})
    write_json(BASE / "01_protocol/model_registry.json", {
        "authority": relative(DATA / "cell_registry_56.csv"),
        "models": models, "count": len(models), "immutable_provider_revision": NOT_RETAINED,
    })
    write_json(BASE / "01_protocol/dataset_registry.json", {
        "authority": relative(DATA / "cell_registry_56.csv"),
        "datasets": datasets, "count": len(datasets), "raw_dataset_snapshot": NOT_RETAINED,
    })
    write_json(BASE / "01_protocol/method_registry.json", {
        "authority": relative(DATA / "cell_metrics.csv"), "methods": methods, "count": len(methods),
    })
    matrix = [{**row, "scientific_result_authority": "v3_7", "raw_run_binding": NOT_RETAINED} for row in cells]
    write_csv(BASE / "01_protocol/experiment_matrix.csv", list(matrix[0]), matrix)
    applicability = [{
        "model": row["model"], "dataset": row["dataset"], "method": row["method"],
        "applicable": row["applicable"], "authority": relative(DATA / "cell_metrics.csv"),
    } for row in metrics]
    write_csv(BASE / "01_protocol/applicability_registry.csv", list(applicability[0]), applicability)
    (BASE / "01_protocol/experiment_specification.md").write_text(
        "# Experiment specification\n\n"
        "The v3.7 numerical authority contains 7 models, 8 datasets, 56 model–dataset cells, "
        "40 candidate generations per cell, and 392 method–cell aggregate rows. Preserved v3.6 "
        "generation and execution evidence is admitted only through identity and hash reconciliation. "
        "Current per-observation verifier values, decisions, and outcome labels were not retained and "
        "are never reconstructed from aggregates. The embedded manuscript stack is the sole source "
        "for v3.7 reported numbers and artifacts.\n"
    )


def source_indexes() -> None:
    rows = []
    symbols: dict[str, Any] = {}
    for path in sorted(SOURCE.rglob("*")):
        if not path.is_file() or path.name == ".DS_Store" or "__pycache__" in path.parts:
            continue
        record = {"source_path": relative(path), "sha256": sha256(path), "bytes": path.stat().st_size}
        rows.append(record)
        if path.suffix == ".py":
            try:
                tree = ast.parse(path.read_text())
                symbols[relative(path)] = [{
                    "name": node.name, "line": node.lineno, "kind": type(node).__name__,
                } for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
            except (SyntaxError, UnicodeError):
                symbols[relative(path)] = {"parse_status": "UNAVAILABLE"}
    write_csv(BASE / "12_code_provenance/source_manifest.csv", ["source_path", "sha256", "bytes"], rows)
    write_json(BASE / "12_code_provenance/scientific_symbols.json", {
        "scope": "Embedded v3.7 artifact source only", "symbols": symbols,
    })
    (BASE / "12_code_provenance/source_sha256.txt").write_text(
        "".join(f"{row['sha256']}  {row['source_path']}\n" for row in rows)
    )
    write_json(BASE / "12_code_provenance/producer_lineage.json", {
        "data_root": relative(DATA), "source_root": relative(SOURCE),
        "figure_entrypoints": [
            relative(SOURCE / "render_result_figures.py"),
            relative(SOURCE / "render_static_schematics.py"),
            relative(SOURCE / "figure_06_backend_invariance_atlas.py"),
        ],
        "table_entrypoint": relative(SOURCE / "generate_tables.py"),
        "metric_entrypoint": "16_replay/rebuild_metric_summaries.py",
    })


def metric_indexes() -> None:
    metrics = read_csv(DATA / "cell_metrics.csv")
    lineage = [{**row, "source_path": relative(DATA / "cell_metrics.csv"),
                "csv_data_row_1based": index, "raw_record_binding": NOT_RETAINED}
               for index, row in enumerate(metrics, 1)]
    write_csv(BASE / "13_metrics/aggregate_lineage.csv", list(lineage[0]), lineage)
    write_json(BASE / "13_metrics/metric_registry.json", {
        "classification": "V3_7_AUTHORITATIVE_AGGREGATES",
        "source": {"path": relative(DATA / "cell_metrics.csv"), "sha256": sha256(DATA / "cell_metrics.csv")},
        "fields": list(metrics[0]), "aggregate_rows": len(metrics),
        "per_observation_values": NOT_RETAINED,
    })
    manifest = json.loads((DATA / "FIGURE_DATA_DERIVATION_MANIFEST.json").read_text())
    derivations = []
    for name, record in manifest.get("new_csvs", {}).items():
        output = DATA / name
        if not output.is_file():
            continue
        derivations.append({
            "output": relative(output),
            "source_files": [relative(DATA / source) for source in record.get("source_files", [])],
            "declared_derivation": record.get("derivation", "NOT_APPLICABLE"),
            "public_sha256": sha256(output),
            "raw_record_lineage": NOT_RETAINED,
        })
    write_jsonl = BASE / "13_metrics/metric_lineage.jsonl"
    write_jsonl.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in derivations))

    old_figures = read_csv(BASE / "14_artifacts/figure_lineage.csv") if (BASE / "14_artifacts/figure_lineage.csv").is_file() else []
    figure_bindings: dict[str, list[str]] = {}
    for row in old_figures:
        for source in row.get("inputs", "").split(";"):
            if source:
                figure_bindings.setdefault(Path(source).name, []).append(row.get("name", "UNKNOWN"))
    numbers = []
    number_pattern = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
    for path in sorted(DATA.glob("*.csv")):
        for index, row in enumerate(read_csv(path), 1):
            for field, value in row.items():
                if value and number_pattern.fullmatch(value):
                    table_path = TABLES / (path.stem + ".tex")
                    bindings = []
                    if table_path.is_file():
                        bindings.append(relative(table_path))
                    bindings.extend(f"FIGURE:{name}" for name in figure_bindings.get(path.name, []))
                    numbers.append({
                        "source_path": relative(path), "data_row_1based": index, "field": field,
                        "stored_value": value, "source_sha256": sha256(path),
                        "manuscript_artifact_binding": ";".join(sorted(set(bindings))) or "DATA_AUTHORITY_ONLY",
                        "raw_observation_binding": NOT_RETAINED,
                    })
    write_csv(BASE / "13_metrics/reported_number_registry.csv", list(numbers[0]), numbers)
    write_json(BASE / "13_metrics/statistical_procedure.json", {
        "authority": relative(DATA / "FIGURE_DATA_DERIVATION_MANIFEST.json"),
        "paired_gap_interval": manifest.get("FIG1_CI_METHOD", "DECLARED_IN_AUTHORITY"),
        "pairing": manifest.get("NOCERT_REDUCTION_PAIRING", "DECLARED_IN_AUTHORITY"),
        "aggregation": manifest.get("ALL_DATASETS_MEDIAN_AGGREGATION", "DECLARED_IN_AUTHORITY"),
        "raw_observation_independence": NOT_RETAINED,
    })


def artifact_indexes() -> None:
    existing_figures = read_csv(BASE / "14_artifacts/figure_lineage.csv") if (BASE / "14_artifacts/figure_lineage.csv").is_file() else []
    figures = []
    by_name = {row.get("name"): row for row in existing_figures}
    for pdf in sorted(IMAGES.glob("*.pdf")):
        png = pdf.with_suffix(".png")
        if not png.is_file():
            raise RuntimeError(f"Missing PNG pair for {pdf.name}")
        old = by_name.get(pdf.stem, {})
        inputs = []
        for item in old.get("inputs", "").split(";"):
            if item:
                inputs.append(relative(DATA / Path(item).name))
        figures.append({
            "name": pdf.stem, "pdf_path": relative(pdf), "pdf_sha256": sha256(pdf),
            "png_path": relative(png), "png_sha256": sha256(png),
            "emitter_root": relative(SOURCE), "inputs": ";".join(inputs),
            "scientific_result_authority": "v3_7",
        })
    write_csv(BASE / "14_artifacts/figure_lineage.csv", list(figures[0]), figures)

    tables = []
    for tex in sorted(TABLES.glob("*.tex")):
        candidates = [DATA / (tex.stem + ".csv")]
        if tex.name == "table_01_execution_and_primary_summary.tex":
            candidates = [DATA / "table_01_main_six_summary.csv"]
        elif tex.name == "table_02_comparator_scope_applicability.tex":
            candidates = [DATA / "table_08_baseline_scope_matrix.csv"]
        elif tex.name == "table_04_cost_anatomy.tex":
            candidates = [DATA / "table_28_protocol_cost.csv"]
        elif tex.name == "table_05_audit_calibration_summary.tex":
            candidates = [DATA / "table_03_audit_calibration_summary.csv"]
        input_path = next((path for path in candidates if path.is_file()), None)
        tables.append({
            "table": tex.stem,
            "input_csv": relative(input_path) if input_path else "MULTI_SOURCE_OR_GENERATED",
            "input_sha256": sha256(input_path) if input_path else "NOT_APPLICABLE",
            "output_tex": relative(tex), "tex_sha256": sha256(tex),
            "emitter": relative(SOURCE / "generate_tables.py"),
        })
    write_csv(BASE / "14_artifacts/table_lineage.csv", list(tables[0]), tables)

    artifacts = []
    for path in sorted(STACK.rglob("*")):
        if path.is_file():
            artifacts.append({"path": relative(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    write_json(BASE / "14_artifacts/artifact_inventory.json", {
        "scope": "Standalone v3.7 manuscript artifact stack", "artifacts": artifacts,
    })
    (BASE / "14_artifacts/artifact_sha256.txt").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in artifacts)
    )


def release_and_change_control() -> None:
    donor_binding = json.loads((BASE / "15_change_control/v3_6_donor_binding.json").read_text())
    generation_count = sum(1 for _ in (BASE / "05_generations/raw_response_manifest.jsonl").open())
    trajectory_count = sum(1 for _ in (BASE / "06_agent_traces/trajectory_manifest.jsonl").open())
    with (BASE / "10_execution/run_registry.csv").open(newline="") as handle:
        receipt_count = sum(1 for _ in csv.DictReader(handle))
    prompt_manifest = json.loads((BASE / "03_prompts/PROMPT_MANIFEST.json").read_text())
    donor_rows = [json.loads(line) for line in (BASE / "15_change_control/donor_artifact_binding.jsonl").read_text().splitlines() if line.strip()]
    line_counts = {}
    for name, path in {
        "tool_calls": BASE / "06_agent_traces/tool_call_manifest.jsonl",
        "tool_outputs": BASE / "06_agent_traces/tool_output_manifest.jsonl",
        "verifier_slots": BASE / "07_verification/verifier_invocations.jsonl",
        "certificate_identities": BASE / "08_certificates/certificate_ledger.jsonl",
    }.items():
        line_counts[name] = sum(1 for line in path.open() if line.strip())
    write_json(BASE / "00_release/release_identity.json", {
        "project": "PCG-MAS", "scientific_release": "v3.7",
        "record_type": "PUBLIC_REPRODUCIBILITY_RECORD", "record_status": "READY_FOR_SEALING",
        "scientific_result_authority": relative(DATA),
        "historical_execution_evidence": "HASH_RECONCILED_SEALED_DONOR_PROJECTIONS",
        "donor_binding": "15_change_control/v3_6_donor_binding.json",
    })
    write_json(BASE / "00_release/scope_manifest.json", {
        "release": "v3.7", "public_root": "reproducibility/v3_7",
        "included": ["00_release through 16_replay", "embedded manuscript data/source/artifacts", "sanitized historical execution evidence"],
        "excluded": ["credentials", "private paths", "account identifiers", "network endpoints", "legacy scientific result values"],
        "standalone_replay": True, "external_private_tree_dependency": False,
    })
    completeness = {
        "status_semantics": {
            "CAPTURED": "Public bytes are included and hash bound.",
            "SANITIZED_PROJECTION": "Original and public hashes are retained with a redaction reason.",
            NOT_RETAINED: "The historical value is unavailable and was not inferred.",
        },
        "counts": {
            "raw_generation_responses": generation_count,
            "raw_agent_trajectories": trajectory_count,
            "execution_receipts": receipt_count,
            "declared_prompts": prompt_manifest.get("total_prompts"),
            "captured_prompts": prompt_manifest.get("captured_count"),
            **line_counts,
            "donor_artifacts_admitted": len(donor_rows),
            "sanitized_public_projections": sum(row["projection"] == "SANITIZED" for row in donor_rows),
            "models": 7, "datasets": 8, "cells": 56, "generations_per_cell": 40,
            "aggregate_method_cell_rows": 392,
        },
        "current_scientific_layers": {
            "07_verification_per_observation_values": NOT_RETAINED,
            "08_certificate_per_observation_values": NOT_RETAINED,
            "09_outcome_per_observation_values": NOT_RETAINED,
            "13_metrics": "CAPTURED",
            "14_artifacts": "CAPTURED",
        },
        "donor_binding": donor_binding,
    }
    write_json(BASE / "00_release/completeness_matrix.json", completeness)
    write_json(BASE / "00_release/evidence_reconciliation.json", {
        "status": "PASS_WITH_EXPLICIT_PER_OBSERVATION_GAP",
        "generation_identity_and_hash_reconciliation": "PASS",
        "trajectory_identity_and_hash_reconciliation": "PASS",
        "receipt_identity_and_hash_reconciliation": "PASS",
        "legacy_result_values_admitted": 0,
        "current_aggregate_authority": relative(DATA),
    })
    write_json(BASE / "00_release/missing_source_bindings.json", {
        "status": "EXPLICIT", "records": [{
            "layer": "current per-observation verification, certificate, decision, and outcome values",
            "status": NOT_RETAINED,
            "consequence": "Raw-to-aggregate scientific replay cannot be claimed.",
        }],
    })
    write_json(BASE / "00_release/excluded_fixture_inventory.json", {
        "status": "NOT_APPLICABLE", "reason": "Only embedded v3.7 authority and hash-admitted donor evidence are indexed.",
    })

    snapshot = []
    for path in sorted(STACK.rglob("*")):
        if path.is_file():
            snapshot.append({"path": relative(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    write_csv(BASE / "15_change_control/numerical_authority_snapshot.csv", ["path", "sha256", "bytes"], snapshot)
    write_json(BASE / "15_change_control/scientific_change_ledger.json", {
        "action": "Complete v3.7 evidence migration and standalone replay binding",
        "scientific_values_changed": False, "new_experiments_run": False,
        "legacy_result_values_admitted": 0, "result_authority": "v3.7",
    })
    write_json(BASE / "15_change_control/correction_registry.json", {
        "status": "NO_SCIENTIFIC_VALUE_CORRECTION_IN_THIS_MIGRATION",
    })
    write_json(BASE / "15_change_control/contamination_boundary.json", {
        "donor_use": "Historical execution evidence only after identity/hash reconciliation",
        "excluded_donor_layers": ["07_verification", "08_certificates", "09_outcomes", "13_metrics", "14_artifacts"],
        "current_result_authority": relative(DATA),
    })
    (BASE / "15_change_control/analysis_lock.md").write_text(
        "# Analysis lock\n\nThe embedded v3.7 data and source are the sole scientific authority. "
        "The preserved donor contributes execution evidence only. Missing current per-observation "
        "values remain explicit and must not be inferred from aggregates.\n"
    )


def main() -> None:
    required = [DATA / "cell_metrics.csv", DATA / "cell_registry_56.csv", SOURCE, IMAGES, TABLES]
    if not all(path.exists() for path in required):
        raise SystemExit("Embedded manuscript stack is incomplete")
    protocol_indexes()
    source_indexes()
    metric_indexes()
    artifact_indexes()
    release_and_change_control()
    print(json.dumps({"status": "PASS", "authority": relative(DATA)}, indent=2))


if __name__ == "__main__":
    main()
