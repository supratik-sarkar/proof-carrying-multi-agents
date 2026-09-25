"""Deterministic derivation of Table 34: Verifiability and Execution Ledger.

Derives the verifiability and execution ledger strictly from upstream authoritative
protocol coverage, manifest, and auditor invariance sources.
"""
from pathlib import Path
import hashlib, os, sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('PCG_MANUSCRIPT_DATA', ROOT / 'data'))

def compute_sha256(filepath):
    return hashlib.sha256(Path(filepath).read_bytes()).hexdigest()

def derive_ledger():
    coverage_path = DATA / 'table_26_protocol_coverage.csv'
    registry_path = DATA / 'cell_registry_56.csv'
    manifest_path = DATA / 'table_29_protocol_manifest.csv'
    agreement_path = DATA / 'table_27_protocol_auditor_invariance.csv'
    output_path = DATA / 'table_34_verifiability_execution_ledger.csv'

    # Upstream source files verification
    for p in [coverage_path, registry_path, manifest_path, agreement_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing upstream source: {p}")

    registry = pd.read_csv(registry_path)
    coverage = pd.read_csv(coverage_path).set_index('Quantity')['Value'].astype(int)
    bindings = pd.read_csv(manifest_path)
    agreement = pd.read_csv(agreement_path, keep_default_na=False)

    # Cross-assertions between coverage and registry
    assert registry.model.nunique() == coverage['Models'] == 7, "Model count mismatch"
    assert registry.dataset.nunique() == coverage['Datasets'] == 8, "Dataset count mismatch"
    assert len(registry) == coverage['Cells'] == 56, "Cell count mismatch"
    assert (registry.generations == coverage['Generations per cell']).all(), "Generations per cell mismatch"
    assert registry.generations.sum() == coverage['Canonical observations'] == 2240, "Canonical observations mismatch"

    bound_generators = len(bindings[(bindings.Role == 'Frozen generator') & (bindings.Status == 'Bound')])
    assert bound_generators == 7, f"Bound generator count mismatch: {bound_generators} != 7"

    rows = [
        {'Quantity': 'Models / datasets / cells', 'Value': f"{coverage['Models']} / {coverage['Datasets']} / {coverage['Cells']}"},
        {'Quantity': 'Generations per cell', 'Value': str(coverage['Generations per cell'])},
        {'Quantity': 'Canonical observations', 'Value': f"{coverage['Canonical observations']:,}"},
        {'Quantity': 'Frozen generator bindings', 'Value': str(bound_generators)},
    ]

    for row in agreement.itertuples(index=False):
        if row.Channel == 'Final Check':
            rows.append({'Quantity': 'Final-check agreement', 'Value': str(row.Agreement)})
            rows.append({'Quantity': 'Final-check disagreements', 'Value': str(row.Disagreements)})
        else:
            rows.append({'Quantity': f'Cross-host {row.Channel} agreement', 'Value': str(row.Agreement)})

    df_ledger = pd.DataFrame(rows)
    assert len(df_ledger) == 10, f"Expected exactly 10 ledger rows, got {len(df_ledger)}"

    # Deterministic write
    df_ledger.to_csv(output_path, index=False, lineterminator='\n', encoding='utf-8')
    sha = compute_sha256(output_path)
    print(f"LEDGER_DERIVATION_SUCCESS=YES ROWS={len(df_ledger)} SHA256={sha}")
    return df_ledger

if __name__ == '__main__':
    derive_ledger()
