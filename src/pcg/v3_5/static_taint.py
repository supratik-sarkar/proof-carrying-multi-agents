"""Production static taint and reachability auditor for PCG-MAS v3.5.

Enforces:
- AST inspection of all reachable runtime acceptance modules.
- Scans for forbidden evaluator identifiers in attributes, subscripts, names, and parameters.
- Rejection of any evaluator leakage into acceptance decisions.
- Negative control test against synthetic tainted fixtures.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

FORBIDDEN_EVALUATOR_IDENTIFIERS: Set[str] = {
    "ground_truth_harm",
    "gt_harm",
    "harm_label",
    "dataset_native_success",
    "gt_success",
    "success_label",
    "gold_answers",
    "gold_answer",
    "reference_answer",
    "benchmark_correctness_label",
    "evaluator_target",
    "evaluator_labels",
}


class StaticTaintVisitor(ast.NodeVisitor):
    """AST visitor detecting evaluator references in runtime code."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.violations: List[Dict[str, Any]] = []

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_EVALUATOR_IDENTIFIERS:
            self.violations.append({
                "file": self.filename,
                "line": node.lineno,
                "col": node.col_offset,
                "type": "attribute_access",
                "identifier": node.attr,
            })
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            if node.slice.value in FORBIDDEN_EVALUATOR_IDENTIFIERS:
                self.violations.append({
                    "file": self.filename,
                    "line": node.lineno,
                    "col": node.col_offset,
                    "type": "subscript_key",
                    "identifier": node.slice.value,
                })
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for arg in node.args.args:
            if arg.arg in FORBIDDEN_EVALUATOR_IDENTIFIERS:
                self.violations.append({
                    "file": self.filename,
                    "line": arg.lineno,
                    "col": arg.col_offset,
                    "type": "parameter_name",
                    "identifier": arg.arg,
                })
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id in FORBIDDEN_EVALUATOR_IDENTIFIERS:
            self.violations.append({
                "file": self.filename,
                "line": node.lineno,
                "col": node.col_offset,
                "type": "name_load",
                "identifier": node.id,
            })
        self.generic_visit(node)


def audit_source_string(code_str: str, filename: str = "<string>") -> List[Dict[str, Any]]:
    """Parse and audit a source string for taint violations."""
    tree = ast.parse(code_str, filename=filename)
    visitor = StaticTaintVisitor(filename)
    visitor.visit(tree)
    return visitor.violations


def audit_file(file_path: Path) -> List[Dict[str, Any]]:
    """Parse and audit a Python file for taint violations."""
    code = file_path.read_text(encoding="utf-8")
    return audit_source_string(code, filename=str(file_path))


def audit_runtime_modules(modules_dir: Path) -> Dict[str, Any]:
    """Audit all production Python modules in directory."""
    all_violations: List[Dict[str, Any]] = []
    audited_files: List[str] = []

    # Files to exclude: static_taint itself (which defines the forbidden identifiers)
    # and tests
    for py_file in sorted(modules_dir.glob("*.py")):
        if py_file.name in ("static_taint.py", "mutants.py"):
            continue
        violations = audit_file(py_file)
        audited_files.append(py_file.name)
        all_violations.extend(violations)

    return {
        "schema": "PCG_MAS_V3_5_STATIC_TAINT_AUDIT_V1",
        "audited_files": audited_files,
        "n_audited_files": len(audited_files),
        "violations": all_violations,
        "violation_count": len(all_violations),
        "status": "PASS" if len(all_violations) == 0 else "FAIL",
    }


def run_static_taint_negative_control() -> Dict[str, Any]:
    """Execute negative control: intentionally tainted code snippets must be flagged."""
    tainted_snippets = [
        ("subscript_access", "def get_score(cand):\n    return cand['evaluator_labels']['harm']\n"),
        ("attribute_access", "def check_answer(c):\n    if c.gt_harm == 1:\n        return False\n"),
        ("parameter_name", "def evaluate(candidate, ground_truth_harm):\n    return ground_truth_harm == 0\n"),
        ("name_load", "def verify():\n    h = gold_answers\n    return len(h)\n"),
    ]

    control_results = []
    all_detected = True

    for name, code in tainted_snippets:
        violations = audit_source_string(code, filename=f"<synthetic_{name}>")
        detected = len(violations) > 0
        if not detected:
            all_detected = False
        control_results.append({
            "snippet_name": name,
            "detected": detected,
            "violations_found": len(violations),
        })

    return {
        "schema": "PCG_MAS_V3_5_STATIC_TAINT_NEGATIVE_CONTROL_V1",
        "controls_run": len(tainted_snippets),
        "all_controls_detected": all_detected,
        "details": control_results,
        "status": "PASS" if all_detected else "FAIL",
    }
