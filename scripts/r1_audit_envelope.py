#!/usr/bin/env python3
"""Compute Theorem 5.1 finite-sample audit envelope from channel counts.

Expected input CSV columns:

    channel,failures,n

where channel is one of:
    int, rep, chk, cov

Example:
    channel,failures,n
    int,2,200
    rep,0,200
    chk,11,200
    cov,8,200

Usage:
    python scripts/r1_audit_envelope.py \
      --input results/r1/channel_counts.csv \
      --output results/r1/audit_envelope.csv \
      --delta 0.05

For a quick local sanity run:
    python scripts/r1_audit_envelope.py --demo
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pcg.eval.audit import AUDIT_CHANNELS, estimate_audit_envelope_from_counts


def _read_counts(path: Path) -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"channel", "failures", "n"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for row in reader:
            channel = row["channel"].strip()
            failures = int(row["failures"])
            n = int(row["n"])
            counts[channel] = (failures, n)

    return counts


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["channel", "n", "failures", "beta_hat", "radius", "U_delta"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/r1/audit_envelope.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("results/r1/audit_envelope.json"))
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        counts = {
            "int": (2, 200),
            "rep": (0, 200),
            "chk": (11, 200),
            "cov": (8, 200),
        }
    else:
        if args.input is None:
            raise SystemExit("Provide --input channel_counts.csv or use --demo.")
        counts = _read_counts(args.input)

    envelope = estimate_audit_envelope_from_counts(
        counts,
        delta=args.delta,
        channels=AUDIT_CHANNELS,
    )

    _write_csv(args.output, envelope.rows())

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(envelope.to_dict(), indent=2), encoding="utf-8")

    print(f"Wrote CSV : {args.output}")
    print(f"Wrote JSON: {args.json_output}")
    print(f"sum_j U_j(delta={args.delta}) = {envelope.sum_U_delta:.6f}")


if __name__ == "__main__":
    main()