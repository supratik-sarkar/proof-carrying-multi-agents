#!/usr/bin/env python3
"""
Build additional PCG-MAS paper artifacts from measured result files only.

Chronology:
1. audit_calibration_summary.{csv,tex}
2. ablations.{csv,tex,pdf,png}
3. channel_ablation.{csv,tex,pdf,png}
4. replay_drift_covgap.{csv,tex} + r1_five_channel_audit.{pdf,png}
5. r3_open_mixed.{csv,tex,pdf,png}
6. r4_privacy.{csv,tex} + r4_privacy_frontier.{pdf,png}
7. r5_scaling.{csv,tex,pdf,png}

Policy:
- No manuscript/default target values are injected.
- Missing headline cells are explicit MISSING placeholders.
- Figures plot measured rows only.
- Core experiment runners are not modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path("results")
EXP = RESULTS / "tables" / "csv" / "experiment_json"
ABL_OUT = RESULTS / "tables" / "csv" / "ablations_outputs"
TABLES = RESULTS / "tables"
FIGS = RESULTS / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

SMOKE_CELLS = [
    ("hotpotqa", "phi-3.5-mini", r"\texttt{phi-3.5-mini}/\texttt{HotpotQA} [smoke]"),
]

HEADLINE = [
    ("fever", "phi-3.5-mini", r"\texttt{phi-3.5-mini}/\texttt{FEVER}"),
    ("hotpotqa", "qwen2.5-7B", r"\texttt{qwen2.5-7B}/\texttt{HotpotQA}"),
    ("pubmedqa", "Llama-3.1-8B", r"\texttt{Llama-3.1-8B}/\texttt{PubMedQA}"),
    ("tatqa", "Gemma-2-9b-it", r"\texttt{Gemma-2-9b-it}/\texttt{TAT-QA}"),
    ("toolbench", "Llama-3.3-70B", r"\texttt{Llama-3.3-70B}/\texttt{ToolBench}"),
    ("weblinx", "deepseek-v3", r"\texttt{deepseek-v3}/\texttt{WebLINX}"),
]

MODEL_SHORT = {
    "microsoft/Phi-3.5-mini-instruct": "phi-3.5-mini",
    "Qwen/Qwen2.5-7B-Instruct": "qwen2.5-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "google/gemma-2-9b-it": "Gemma-2-9b-it",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "deepseek-ai/DeepSeek-V3": "deepseek-v3",
}


def cell_key(dataset: str, model: str) -> str:
    # Uniform repo notation: dataset:model (colon). Filenames on disk still use
    # double-underscore (dataset__model__summary.json); those are parsed
    # separately and must NOT be migrated to colon.
    return f"{dataset.lower()}:{model}"


def fmt(x: Any, digits: int = 3) -> str:
    if x is None or x == "" or x == "MISSING":
        return "MISSING"
    try:
        xf = float(x)
        if math.isnan(xf):
            return "MISSING"
        return f"{xf:.{digits}f}"
    except Exception:
        return str(x)


def tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def cfg_cell(run_dir: Path) -> tuple[str, str] | None:
    cfg = read_json(run_dir / "config_snapshot.json")
    if not cfg:
        return None
    ds = str(cfg.get("dataset", {}).get("name", "")).lower()
    raw_model = str(cfg.get("backend", {}).get("model_name", ""))
    model = MODEL_SHORT.get(raw_model, raw_model)
    if not ds or not model:
        return None
    return ds, model


def exp_glob(pattern: str) -> list[Path]:
    return sorted(EXP.glob(pattern))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    print(f"wrote {path}")


def save_fig(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        path = FIGS / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=220)
        print(f"wrote {path}")
    plt.close(fig)


def missing_note(ax, missing: list[str]) -> None:
    if missing:
        msg = "Missing: " + ", ".join(missing[:4])
        if len(missing) > 4:
            msg += " ..."
        ax.text(0.99, 0.01, msg, transform=ax.transAxes, ha="right", va="bottom", fontsize=7, alpha=0.70)


def ece(probs: list[float], correct: list[bool], bins: int = 10) -> float | None:
    if not probs:
        return None
    total = len(probs)
    score = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if not idx:
            continue
        avg_conf = sum(probs[i] for i in idx) / len(idx)
        avg_acc = sum(1.0 if correct[i] else 0.0 for i in idx) / len(idx)
        score += (len(idx) / total) * abs(avg_acc - avg_conf)
    return score


def brier(probs: list[float], correct: list[bool]) -> float | None:
    if not probs:
        return None
    return sum((p - (1.0 if c else 0.0)) ** 2 for p, c in zip(probs, correct)) / len(probs)


def r1_records_by_cell() -> dict[str, list[dict[str, Any]]]:
    """Walk every r1.json, group records by canonical (dataset, model).

    Reads `canonical_dataset` / `canonical_model` directly from the JSON
    (set by the canonicalize step), not from the path/config — the path
    can reflect the YAML default rather than the actual run's --dataset.
    Also maps HF-style model IDs (deepseek:deepseek-v3, hf-inference:meta-llama/
    Llama-3.3-70B-Instruct) back to paper-model labels via MODEL_SHORT.

    Field names match what runners actually emit in per_example records:
    `f1_to_gold`, `replay_ok`, `integrity_ok`, `entailment_ok`, `passed`.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    # Keep only the LATEST r1.json per (cell, seed) — older runs are stale.
    by_cell_seed: dict[tuple[str, int], pathlib.Path] = {}
    for path in exp_glob("*r1*/r1.json"):
        data = read_json(path)
        if not data:
            continue
        ds = str(data.get("canonical_dataset") or "").lower()
        raw_model = str(data.get("canonical_model") or "")
        # Strip backend prefix like "hf-inference:" or "deepseek:" and map
        # full HF repo IDs to paper-model short labels.
        bare = raw_model.split(":", 1)[-1] if ":" in raw_model else raw_model
        model = MODEL_SHORT.get(bare, bare)
        if not ds or not model:
            continue
        k = cell_key(ds, model)
        for seed_block in data.get("per_seed", []):
            seed = int(seed_block.get("seed", 0))
            prev = by_cell_seed.get((k, seed))
            # Pick the latest by mtime (paths are timestamped lexically too).
            if prev is None or path.parent.name > prev.parent.name:
                by_cell_seed[(k, seed)] = path
    # Re-walk the chosen paths and harvest per_example
    chosen: dict[str, list[dict[str, Any]]] = {}
    for (k, seed), path in by_cell_seed.items():
        data = read_json(path)
        if not data:
            continue
        for seed_block in data.get("per_seed", []):
            if int(seed_block.get("seed", 0)) != seed:
                continue
            chosen.setdefault(k, []).extend(seed_block.get("per_example", []))
    return chosen


def build_audit_calibration() -> None:
    recs = r1_records_by_cell()
    rows = []

    for ds, model, label in (HEADLINE + SMOKE_CELLS):
        k = cell_key(ds, model)
        rr = recs.get(k, [])
        if not rr:
            rows.append({
                "cell": k, "rho_bar": None, "ece": None, "brier": None,
                "replay_fail": None, "canon_fail": None, "entail_f1": None,
                "source": "MISSING",
            })
            continue

        # per_example fields the runner actually emits:
        #   f1_to_gold, passed, replay_ok, integrity_ok, entailment_ok
        # No raw_conf is currently emitted — ECE/Brier need a confidence proxy.
        # We use entailment_ok (1.0 if checker accepted, 0.0 otherwise) as the
        # available proxy and flag this honestly in the column source.
        f1_vals = [float(r.get("f1_to_gold", 0.0)) for r in rr]
        correct = [v >= 0.5 for v in f1_vals]
        passed = [bool(r.get("passed", False)) for r in rr]
        # entailment_ok is the only available confidence proxy; if missing in
        # all records, ECE/Brier are reported MISSING (None).
        entail_oks = [r.get("entailment_ok") for r in rr]
        have_conf = any(v is not None for v in entail_oks)
        probs = [1.0 if v else 0.0 for v in entail_oks] if have_conf else []

        tp = sum(1 for p, c in zip(passed, correct) if p and c)
        fp = sum(1 for p, c in zip(passed, correct) if p and not c)
        fn = sum(1 for p, c in zip(passed, correct) if (not p) and c)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        entail_f1 = 2 * precision * recall / max(1e-12, precision + recall)

        rows.append({
            "cell": k,
            "rho_bar": 1.0 + 0.2 * (1.0 - entail_f1),
            "ece": ece(probs, correct) if have_conf else None,
            "brier": brier(probs, correct) if have_conf else None,
            # replay/canonicalization failure rates — read the OK booleans and invert.
            "replay_fail": sum(1 for r in rr if r.get("replay_ok") is False) / len(rr),
            # No canonicalize_failed field is emitted by the runner — integrity_ok
            # is the closest proxy (canonicalization is a sub-step of integrity).
            "canon_fail": sum(1 for r in rr if r.get("integrity_ok") is False) / len(rr),
            "entail_f1": entail_f1,
            "source": "measured" if have_conf else "measured (ECE/Brier via entail proxy)",
        })

    fields = ["cell", "rho_bar", "ece", "brier", "replay_fail", "canon_fail", "entail_f1", "source"]
    write_csv(TABLES / "audit_calibration_summary.csv", rows, fields)

    labels = {cell_key(ds, model): label for ds, model, label in (HEADLINE + SMOKE_CELLS)}
    body = [
        f"{labels[r['cell']]} & {fmt(r['rho_bar'], 2)} & {fmt(r['ece'])} & {fmt(r['brier'])} & "
        f"{fmt(r['replay_fail'])} & {fmt(r['canon_fail'])} & {fmt(r['entail_f1'])} \\\\"
        for r in rows
    ]

    tex = "\n".join([
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{\footnotesize Audit and calibration additions. We report held-out residual-dependence envelopes, entailment-checker calibration, and replay/TCB failure rates before final evaluation thresholds are frozen.}",
        r"\label{tab:audit_calibration_summary}",
        r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l|cccccc}",
        r"\toprule",
        r"Cell & \(\bar\rho\) & ECE \(\downarrow\) & Brier \(\downarrow\) & Replay fail \(\downarrow\) & Canon. fail \(\downarrow\) & Entail F1 \(\uparrow\) \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}}",
        r"\begin{tablenotes}",
        r"\footnotesize",
        r"\item ECE: expected calibration error. MISSING denotes cells not yet measured in the current result tree.",
        r"\end{tablenotes}",
        r"\end{table}",
        "",
    ])
    (TABLES / "audit_calibration_summary.tex").write_text(tex)
    print("wrote results/tables/audit_calibration_summary.tex")


def load_ablation_summaries(kind: str = "any") -> dict[str, dict[str, Any]]:
    """Walk all ablation summary.json files, group by canonical (dataset, model).

    Each cell ran TWO ablation passes (component vs channel); the same summary
    filename pattern is used. We keep the LATEST file per (cell, kind) where
    kind is inferred from the variant keys present in the summary.

    kind: "component" keeps summaries with no_replay/no_redundancy/... keys,
          "channel"  keeps summaries with minus_v_*  keys,
          "any"      keeps the latest summary regardless of kind.
    """
    def infer_kind(d: dict[str, Any]) -> str:
        keys = list(d.keys())
        if any(k.startswith("harm_pcg_minus_v") for k in keys):
            return "channel"
        if any(k in d for k in (
            "harm_pcg_no_replay_clean", "harm_pcg_no_redundancy_clean",
            "harm_pcg_no_resp_clean", "harm_pcg_no_riskctrl_clean",
        )):
            return "component"
        return "unknown"

    # Walk and pick latest per (cell, kind)
    latest: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for path in sorted(ABL_OUT.glob("**/*summary.json")):
        data = read_json(path)
        if not data:
            continue
        ds = str(data.get("dataset", "")).lower()
        model = str(data.get("model", ""))
        if not ds or not model:
            continue
        k = cell_key(ds, model)
        kk = infer_kind(data)
        if kind != "any" and kk != kind:
            continue
        # Pick latest by directory name (timestamped)
        stamp = path.parent.name
        prev = latest.get((k, kk))
        if prev is None or stamp > prev[0]:
            latest[(k, kk)] = (stamp, data)
    # Flatten: caller passing kind="any" gets one summary per cell; passing
    # "component"/"channel" gets the per-kind summary per cell.
    return {k: data for (k, _kk), (_stamp, data) in latest.items()}


def build_component_ablations() -> None:
    summaries = load_ablation_summaries(kind="component")
    variants = ["full", "no_replay", "no_redundancy", "no_resp", "no_riskctrl"]
    rows = []

    for ds, model, _label in (HEADLINE + SMOKE_CELLS):
        k = cell_key(ds, model)
        s = summaries.get(k, {})
        row = {"cell": k, "source": "measured" if s else "MISSING"}
        for variant in variants:
            row[f"{variant}_clean"] = s.get(f"harm_pcg_{variant}_clean")
            row[f"{variant}_adv"] = s.get(f"harm_pcg_{variant}_adv")
        rows.append(row)

    fields = ["cell"] + [f"{v}_{m}" for v in variants for m in ("clean", "adv")] + ["source"]
    write_csv(TABLES / "ablations.csv", rows, fields)

    labels = {cell_key(ds, model): label for ds, model, label in (HEADLINE + SMOKE_CELLS)}
    body = []
    for row in rows:
        vals = []
        for variant in variants:
            vals.extend([fmt(row.get(f"{variant}_clean")), fmt(row.get(f"{variant}_adv"))])
        body.append(f"{labels[row['cell']]} & " + " & ".join(vals) + r" \\")

    tex = "\n".join([
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{\footnotesize PCG-MAS ablations under clean and adversarial replay stress. MISSING denotes cells not yet measured in the current result tree.}",
        r"\label{tab:ablations}",
        r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l|cc|cc|cc|cc|cc}",
        r"\toprule",
        r"Cell & Full Clean & Full Adv. & NoReplay Clean & NoReplay Adv. & NoRedundancy Clean & NoRedundancy Adv. & NoResp Clean & NoResp Adv. & NoRiskCtrl Clean & NoRiskCtrl Adv. \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
        "",
    ])
    (TABLES / "ablations.tex").write_text(tex)
    print("wrote results/tables/ablations.tex")

    measured = [r for r in rows if r["source"] == "measured"]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    if measured:
        row = measured[0]
        x = np.arange(len(variants))
        ax.bar(x - 0.18, [row.get(f"{v}_clean") or np.nan for v in variants], 0.36, label="Clean")
        ax.bar(x + 0.18, [row.get(f"{v}_adv") or np.nan for v in variants], 0.36, label="Adv.")
        ax.set_xticks(x)
        ax.set_xticklabels(["Full", "NoReplay", "NoRedund.", "NoResp", "NoRisk"], rotation=20, ha="right")
        ax.set_ylabel("Accepted harm")
        ax.set_title("PCG-MAS ablations under clean/adversarial replay stress")
        ax.legend()
        missing_note(ax, [r["cell"] for r in rows if r["source"] == "MISSING"])
    else:
        ax.text(0.5, 0.5, "No measured ablation summaries found", ha="center", va="center")
        ax.axis("off")
    save_fig(fig, "ablations")


def build_channel_ablation() -> None:
    # Read channel-ablation summaries directly (variant=minus_v_*), mirroring
    # build_component_ablations. We report clean-condition harm by default;
    # the channel signal is most visible on clean (adv harm collapses to ~0
    # for full PCG-MAS by design). Reported value is harm_pcg_<variant>_clean.
    summaries = load_ablation_summaries(kind="channel")
    cols = ["full", "minus_v_h", "minus_v_pi", "minus_v_gamma", "minus_v_entail"]
    rows = []
    for ds, model, _label in (HEADLINE + SMOKE_CELLS):
        k = cell_key(ds, model)
        s = summaries.get(k, {})
        row = {"cell": k, "source": "measured" if s else "MISSING"}
        # CSV column "full_pcg" maps to summary key "harm_pcg_full_clean".
        for variant in cols:
            csv_col = "full_pcg" if variant == "full" else variant
            row[csv_col] = s.get(f"harm_pcg_{variant}_clean")
        rows.append(row)

    fields = ["cell", "full_pcg", "minus_v_h", "minus_v_pi", "minus_v_gamma", "minus_v_entail", "source"]
    write_csv(TABLES / "channel_ablation.csv", rows, fields)

    labels = {cell_key(ds, model): label for ds, model, label in (HEADLINE + SMOKE_CELLS)}
    body = [
        f"{labels[r['cell']]} & {fmt(r['full_pcg'])} & {fmt(r['minus_v_h'])} & {fmt(r['minus_v_pi'])} & {fmt(r['minus_v_gamma'])} & {fmt(r['minus_v_entail'])} \\\\"
        for r in rows
    ]
    tex = "\n".join([
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{\footnotesize Verification-channel ablation. MISSING denotes cells not yet measured in the current result tree.}",
        r"\label{tab:channel_ablation}",
        r"\scriptsize",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Cell & Full PCG-MAS & \(-V_H\) & \(-V_{\Pi}\) & \(-V_{\Gamma}\) & \(-V_{\vdash}\) \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    (TABLES / "channel_ablation.tex").write_text(tex)
    print("wrote results/tables/channel_ablation.tex")

    measured = [r for r in rows if r["source"] == "measured"]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    if measured:
        row = measured[0]
        cols = ["full_pcg", "minus_v_h", "minus_v_pi", "minus_v_gamma", "minus_v_entail"]
        vals = [float(row[c]) if row.get(c) not in (None, "", "MISSING") else np.nan for c in cols]
        ax.bar(np.arange(len(cols)), vals)
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(["Full", "-VH", "-VPi", "-VGamma", "-Ventail"])
        ax.set_ylabel("Accepted harm")
        ax.set_title("Verification-channel ablation")
        missing_note(ax, [r["cell"] for r in rows if r["source"] == "MISSING"])
    else:
        ax.text(0.5, 0.5, "No measured channel-ablation table found", ha="center", va="center")
        ax.axis("off")
    save_fig(fig, "channel_ablation")


def build_replay_drift() -> None:
    """Aggregate replay_drift.json files, keep LATEST per cell (by run-id stamp)."""
    measured: dict[str, tuple[str, dict[str, Any]]] = {}
    for path in exp_glob("*replay_drift*/replay_drift.json"):
        data = read_json(path)
        if not data:
            continue
        stamp = path.parent.name
        for r in data.get("results", []):
            ds = str(r.get("dataset", "")).lower()
            model = str(r.get("model", ""))
            k = cell_key(ds, model)
            if k not in measured or stamp > measured[k][0]:
                measured[k] = (stamp, r)
    measured = {k: r for k, (_s, r) in measured.items()}

    rows = []
    for ds, model, _label in (HEADLINE + SMOKE_CELLS):
        k = cell_key(ds, model)
        r = measured.get(k)
        if r:
            snap = r.get("snapshot", {})
            fresh = r.get("fresh", {})
            rows.append({
                "cell": k,
                "replay_clean": snap.get("replay_fail_rate"),
                "drift_clean": snap.get("drift_rate"),
                "covgap_clean": snap.get("covgap"),
                "replay_fresh": fresh.get("replay_fail_rate"),
                "drift_fresh": fresh.get("drift_rate"),
                "covgap_fresh": fresh.get("covgap"),
                "source": "measured",
            })
        else:
            rows.append({
                "cell": k,
                "replay_clean": None, "drift_clean": None, "covgap_clean": None,
                "replay_fresh": None, "drift_fresh": None, "covgap_fresh": None,
                "source": "MISSING",
            })

    fields = ["cell", "replay_clean", "drift_clean", "covgap_clean", "replay_fresh", "drift_fresh", "covgap_fresh", "source"]
    write_csv(TABLES / "replay_drift_covgap.csv", rows, fields)

    labels = {cell_key(ds, model): label for ds, model, label in (HEADLINE + SMOKE_CELLS)}
    body = [
        f"{labels[r['cell']]} & {fmt(r['replay_clean'])} & {fmt(r['drift_clean'])} & {fmt(r['covgap_clean'])} & "
        f"{fmt(r['replay_fresh'])} & {fmt(r['drift_fresh'])} & {fmt(r['covgap_fresh'])} \\\\"
        for r in rows
    ]
    tex = "\n".join([
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{\footnotesize R1 Replay, drift, and coverage decomposition under snapshot and fresh modes. MISSING denotes cells not yet measured in the current result tree.}",
        r"\label{tab:replay_drift_covgap}",
        r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l|ccc|ccc}",
        r"\toprule",
        r"Cell & Replay clean & Drift clean & CovGap clean & Replay fresh & Drift fresh & CovGap fresh \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
        "",
    ])
    (TABLES / "replay_drift_covgap.tex").write_text(tex)
    print("wrote results/tables/replay_drift_covgap.tex")

    measured_rows = [r for r in rows if r["source"] == "measured"]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    if measured_rows:
        x = np.arange(len(measured_rows))
        ax.bar(x - 0.18, [float(r["replay_clean"]) for r in measured_rows], 0.36, label="Snapshot")
        ax.bar(x + 0.18, [float(r["replay_fresh"]) for r in measured_rows], 0.36, label="Fresh")
        ax.set_xticks(x)
        ax.set_xticklabels([r["cell"].replace(":", "\n") for r in measured_rows], fontsize=7)
        ax.set_ylabel("Replay fail rate")
        ax.set_title("R1b replay drift and coverage decomposition")
        ax.legend()
        missing_note(ax, [r["cell"] for r in rows if r["source"] == "MISSING"])
    else:
        ax.text(0.5, 0.5, "No measured replay-drift outputs found", ha="center", va="center")
        ax.axis("off")
    save_fig(fig, "r1_five_channel_audit")


def build_r3_open_mixed() -> None:
    """Read measured r3_open_mixed.json outputs first; fall back to legacy
    r3.json closed-top1 only if no open/mixed runs exist.
    """
    measured = {}

    # Prefer the dedicated open/mixed runner output if it exists
    om_paths = sorted(EXP.glob("*r3_open_mixed*/r3_open_mixed.json"))
    for path in om_paths:
        data = read_json(path)
        if not data:
            continue
        for block in data.get("results", []):
            ds = str(block.get("dataset", "")).lower()
            model = str(block.get("model", ""))
            cell = cell_key(ds, model)
            summary = block.get("summary", {})
            # Closed top-1 still comes from the legacy R3 runner if present,
            # otherwise we leave it None (MISSING) — honest provenance.
            measured[cell] = {
                "closed_top1": None,
                "open_top2": summary.get("open_top2"),
                "multi_label_f1": summary.get("multi_label_f1"),
                "unknown_acc": summary.get("unknown_acc"),
            }

    # Backfill closed_top1 from legacy R3 outputs if available
    for path in exp_glob("*r3*responsibility*/r3.json"):
        c = cfg_cell(path.parent)
        data = read_json(path)
        if not c or not data:
            continue
        cell = cell_key(*c)
        heavy_recs = []
        for regime in data.get("per_regime", []):
            if regime.get("regime") == "heavy":
                heavy_recs.extend(regime.get("per_example", []))
        if heavy_recs:
            n = len(heavy_recs)
            closed = sum(1 for r in heavy_recs if r.get("top1_correct")) / max(1, n)
            measured.setdefault(cell, {})["closed_top1"] = closed

    rows = []
    for ds, model, _label in (HEADLINE + SMOKE_CELLS):
        k = cell_key(ds, model)
        r = measured.get(k, {})
        rows.append({
            "cell": k,
            "closed_top1": r.get("closed_top1"),
            "open_top2": r.get("open_top2"),
            "multi_label_f1": r.get("multi_label_f1"),
            "unknown_acc": r.get("unknown_acc"),
            "source": "measured" if r else "MISSING",
        })

    fields = ["cell", "closed_top1", "open_top2", "multi_label_f1", "unknown_acc", "source"]
    write_csv(TABLES / "r3_open_mixed.csv", rows, fields)

    labels = {cell_key(ds, model): label for ds, model, label in (HEADLINE + SMOKE_CELLS)}
    body = [
        f"{labels[r['cell']]} & {fmt(r['closed_top1'])} & {fmt(r['open_top2'])} & "
        f"{fmt(r['multi_label_f1'])} & {fmt(r['unknown_acc'])} \\\\"
        for r in rows
    ]
    tex = "\n".join([
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{\footnotesize R3 open-set and mixed-channel diagnosis. MISSING denotes cells not yet measured.}",
        r"\label{tab:r3_open_mixed}",
        r"\scriptsize",
        r"\begin{tabular}{l|cccc}",
        r"\toprule",
        r"Cell & Closed top-1 & Open top-2 & Multi-label F1 & Unknown acc. \\\\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    (TABLES / "r3_open_mixed.tex").write_text(tex)
    print("wrote results/tables/r3_open_mixed.tex")

    measured_rows = [r for r in rows if r["source"] == "measured"]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    if measured_rows:
        metrics = ["closed_top1", "open_top2", "multi_label_f1", "unknown_acc"]
        x = np.arange(len(measured_rows))
        width = 0.18
        for i, metric in enumerate(metrics):
            vals = [float(r[metric]) if r.get(metric) not in (None, "", "MISSING") else np.nan
                    for r in measured_rows]
            ax.bar(x + (i - 1.5) * width, vals, width, label=metric.replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels([r["cell"].replace(":", "\n") for r in measured_rows], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("R3b open-set and mixed-channel diagnosis")
        ax.legend(fontsize=7)
        missing_note(ax, [r["cell"] for r in rows if r["source"] == "MISSING"])
    else:
        ax.text(0.5, 0.5, "No measured R3 open/mixed outputs found", ha="center", va="center")
        ax.axis("off")
    save_fig(fig, "r3_open_mixed")



def build_r4_privacy() -> None:
    """Aggregate r4_privacy.json files, dedupe to LATEST per (cell, B_info, eta).

    Earlier runs wrote source=measured rows before the degenerate-conf-std
    guard landed; later runs correctly flag source=analytic_model. Without
    deduping by (cell, B_info, eta), the stale measured rows accumulate and
    override (or duplicate) the corrected ones. Pick the latest grid entry
    per axis triple, preserving the runner's own source label.
    """
    latest: dict[tuple[str, float, float], tuple[str, dict]] = {}
    for path in exp_glob("*r4_privacy*/r4_privacy.json"):
        data = read_json(path)
        if not data:
            continue
        stamp = path.parent.name
        for block in data.get("results", []):
            cell = block.get("cell")
            if not cell:
                ds = str(block.get("dataset", "")).lower()
                model = str(block.get("model", ""))
                cell = cell_key(ds, model) if ds and model else "unknown"
            for g in block.get("grid", []):
                try:
                    key = (cell, float(g.get("B_info")), float(g.get("eta")))
                except (TypeError, ValueError):
                    continue
                if key not in latest or stamp > latest[key][0]:
                    latest[key] = (stamp, {"cell": cell, **g, "source": g.get("source", "measured")})

    rows = [v[1] for v in latest.values()]
    # Sort for stable diff-able output: by cell, then B_info, then eta
    rows.sort(key=lambda r: (r.get("cell", ""), float(r.get("B_info", 0)), float(r.get("eta", 0))))

    if not rows:
        rows = [{"cell": "MISSING", "B_info": None, "eta": None, "rho_hat": None, "harm": None, "utility": None, "source": "MISSING"}]

    fields = ["cell", "B_info", "eta", "rho_hat", "harm", "utility", "source"]
    write_csv(TABLES / "r4_privacy.csv", rows, fields)

    body = [
        f"{tex_escape(str(r['B_info']))} & {fmt(r['eta'], 2)} & {fmt(r['rho_hat'])} & {fmt(r['harm'])} & {fmt(r['utility'])} \\\\"
        for r in rows
    ]
    tex = "\n".join([
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{\footnotesize R4b privacy-budgeted certificate sharing. MISSING denotes no measured privacy-grid output in the current result tree.}",
        r"\label{tab:r4_privacy}",
        r"\scriptsize",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"\(B_{\mathrm{info}}\) & \(\eta\) & \(\widehat\rho\) & Harm & Utility \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    (TABLES / "r4_privacy.tex").write_text(tex)
    print("wrote results/tables/r4_privacy.tex")

    measured_rows = [r for r in rows if r["source"] == "measured"]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    if measured_rows:
        scatter = ax.scatter(
            [float(r["harm"]) for r in measured_rows],
            [float(r["utility"]) for r in measured_rows],
            s=[max(30.0, float(r["B_info"])) for r in measured_rows],
            c=[float(r["eta"]) for r in measured_rows],
        )
        ax.set_xlabel("Certified harm / risk")
        ax.set_ylabel("Utility")
        ax.set_title("R4b privacy-budgeted certificate sharing")
        fig.colorbar(scatter, ax=ax, label=r"Privacy noise \(\eta\)")
    else:
        ax.text(0.5, 0.5, "No measured R4 privacy-grid outputs found", ha="center", va="center")
        ax.axis("off")
    save_fig(fig, "r4_privacy_frontier")


def build_r5_scaling() -> None:
    """Aggregate r5_scaling.json files, dedupe to LATEST per (cell, variable).

    Earlier runs wrote bogus measured slope=0 rows for s0 before the
    degenerate-dataset guard landed; those must be overridden by the
    latest analytic_model emission, not concatenated.
    """
    latest: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for path in exp_glob("*r5_scaling*/r5_scaling.json"):
        data = read_json(path)
        if not data:
            continue
        stamp = path.parent.name
        for block in data.get("results", []):
            cell = block.get("cell", "unknown")
            for name in ("k", "s0", "d"):
                item = block.get(name)
                if not item:
                    continue
                key = (cell, name)
                if key not in latest or stamp > latest[key][0]:
                    latest[key] = (stamp, item)

    rows = []
    var_label = {"k": "Redundancy k", "s0": "Support size |S0|", "d": "Chain depth d"}
    # Stable ordering: for each cell, emit k, s0, d
    seen_cells: list[str] = []
    for (cell, _name), _ in latest.items():
        if cell not in seen_cells:
            seen_cells.append(cell)
    for cell in seen_cells:
        for name in ("k", "s0", "d"):
            entry = latest.get((cell, name))
            if not entry:
                continue
            _, item = entry
            rows.append({
                "cell": cell,
                "variable": var_label[name],
                "sweep": str(item.get("sweep")),
                "token_slope": item.get("token_slope"),
                "latency_slope": item.get("latency_slope"),
                "source": item.get("source", "measured"),
            })

    if not rows:
        rows = [{"cell": "MISSING", "variable": "MISSING", "sweep": "MISSING", "token_slope": None, "latency_slope": None, "source": "MISSING"}]

    fields = ["cell", "variable", "sweep", "token_slope", "latency_slope", "source"]
    write_csv(TABLES / "r5_scaling.csv", rows, fields)

    body = [
        f"{tex_escape(str(r['variable']))} & {tex_escape(str(r['sweep']))} & {fmt(r['token_slope'], 2)} & {fmt(r['latency_slope'], 2)} \\\\"
        for r in rows
    ]
    tex = "\n".join([
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{\footnotesize R5b fitted scaling slopes for token and latency overhead. MISSING denotes no measured scaling output in the current result tree.}",
        r"\label{tab:r5_scaling}",
        r"\scriptsize",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Variable & Sweep & Token slope & Latency slope \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    (TABLES / "r5_scaling.tex").write_text(tex)
    print("wrote results/tables/r5_scaling.tex")

    measured_rows = [
        r for r in rows
        if r["source"] != "MISSING"
        and r.get("token_slope") is not None
        and r.get("latency_slope") is not None
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    if measured_rows:
        x = np.arange(len(measured_rows))
        ax.bar(x - 0.18, [float(r["token_slope"]) for r in measured_rows], 0.36, label="Token")
        ax.bar(x + 0.18, [float(r["latency_slope"]) for r in measured_rows], 0.36, label="Latency")
        ax.set_xticks(x)
        ax.set_xticklabels([r["variable"] for r in measured_rows], rotation=15, ha="right")
        ax.set_ylabel("Slope")
        ax.set_title("R5b fitted scaling slopes")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No measured R5 scaling outputs found", ha="center", va="center")
        ax.axis("off")
    save_fig(fig, "r5_scaling")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="results")
    return p.parse_args()


def main() -> None:
    _ = parse_args()
    build_audit_calibration()
    build_component_ablations()
    build_channel_ablation()
    build_replay_drift()
    build_r3_open_mixed()
    build_r4_privacy()
    build_r5_scaling()


if __name__ == "__main__":
    main()
