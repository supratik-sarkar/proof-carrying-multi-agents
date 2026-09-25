"""Derive backend invariance atlas dataset for Figure 06.

Derives hierarchical invariance strata across models (Panel A), datasets (Panel B),
and ordered cells (Panel C) strictly from authoritative data/paired_cell_effects.csv.
Adheres to the descriptive observed extrema specification without normal-approx CIs.
"""
from pathlib import Path
import hashlib, json, os, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('PCG_MANUSCRIPT_DATA', ROOT / 'data'))

def compute_sha256(filepath):
    return hashlib.sha256(Path(filepath).read_bytes()).hexdigest()

def derive_strata(df):
    # Ensure display quantity exists: pcg_risk_advantage_pp = -delta_pcg_minus_best_pp
    df = df.copy()
    df['pcg_risk_advantage_pp'] = -df['delta_pcg_minus_best_pp']

    # Panel A: 7 rows, one per model, ordered by pcg_risk_advantage_pp ascending
    rows_a = []
    for model, grp in df.groupby('model'):
        n = len(grp)
        deltas_raw = grp['delta_pcg_minus_best_pp']
        adv = grp['pcg_risk_advantage_pp']
        mean_raw = float(deltas_raw.mean())
        mean_adv = float(adv.mean())
        obs_min = float(adv.min())
        obs_max = float(adv.max())
        min_idx = adv.idxmin()
        min_row = grp.loc[min_idx]
        rows_a.append({
            'panel': 'A',
            'stratum_type': 'MODEL',
            'stratum_label': model,
            'n_cells': int(n),
            'delta_pcg_minus_best_pp': mean_raw,
            'pcg_risk_advantage_pp': mean_adv,
            'observed_min_pp': obs_min,
            'observed_max_pp': obs_max,
            'favourable_cells': int((adv > 0).sum()),
            'unfavourable_cells': int((adv < 0).sum()),
            'tie_cells': int((adv == 0).sum()),
            'smallest_advantage_cell_id': str(min_row['cell_id']),
            'smallest_advantage_pp': float(min_row['pcg_risk_advantage_pp']),
            'best_applicable_method': str(min_row['best_applicable_method']),
            'sort_rank': -1
        })
    df_a = pd.DataFrame(rows_a).sort_values('pcg_risk_advantage_pp', ascending=True).reset_index(drop=True)

    # Panel B: 8 rows, one per dataset, ordered by pcg_risk_advantage_pp ascending
    rows_b = []
    for dataset, grp in df.groupby('dataset'):
        n = len(grp)
        deltas_raw = grp['delta_pcg_minus_best_pp']
        adv = grp['pcg_risk_advantage_pp']
        mean_raw = float(deltas_raw.mean())
        mean_adv = float(adv.mean())
        obs_min = float(adv.min())
        obs_max = float(adv.max())
        min_idx = adv.idxmin()
        min_row = grp.loc[min_idx]
        rows_b.append({
            'panel': 'B',
            'stratum_type': 'DATASET',
            'stratum_label': dataset,
            'n_cells': int(n),
            'delta_pcg_minus_best_pp': mean_raw,
            'pcg_risk_advantage_pp': mean_adv,
            'observed_min_pp': obs_min,
            'observed_max_pp': obs_max,
            'favourable_cells': int((adv > 0).sum()),
            'unfavourable_cells': int((adv < 0).sum()),
            'tie_cells': int((adv == 0).sum()),
            'smallest_advantage_cell_id': str(min_row['cell_id']),
            'smallest_advantage_pp': float(min_row['pcg_risk_advantage_pp']),
            'best_applicable_method': str(min_row['best_applicable_method']),
            'sort_rank': -1
        })
    df_b = pd.DataFrame(rows_b).sort_values('pcg_risk_advantage_pp', ascending=True).reset_index(drop=True)

    # Panel C: 56 rows, one per cell, ordered ascending by pcg_risk_advantage_pp (rank 0 to 55)
    df_sorted = df.sort_values('pcg_risk_advantage_pp', ascending=True).reset_index(drop=True)
    rows_c = []
    for rank, row in df_sorted.iterrows():
        raw_d = float(row['delta_pcg_minus_best_pp'])
        adv_d = float(row['pcg_risk_advantage_pp'])
        rows_c.append({
            'panel': 'C',
            'stratum_type': 'CELL',
            'stratum_label': str(row['cell_id']),
            'n_cells': 1,
            'delta_pcg_minus_best_pp': raw_d,
            'pcg_risk_advantage_pp': adv_d,
            'observed_min_pp': adv_d,
            'observed_max_pp': adv_d,
            'favourable_cells': 1 if adv_d > 0 else 0,
            'unfavourable_cells': 1 if adv_d < 0 else 0,
            'tie_cells': 1 if adv_d == 0 else 0,
            'smallest_advantage_cell_id': str(row['cell_id']),
            'smallest_advantage_pp': adv_d,
            'best_applicable_method': str(row['best_applicable_method']),
            'sort_rank': int(rank)
        })
    df_c = pd.DataFrame(rows_c)

    columns = [
        'panel', 'stratum_type', 'stratum_label', 'n_cells',
        'delta_pcg_minus_best_pp', 'pcg_risk_advantage_pp',
        'observed_min_pp', 'observed_max_pp',
        'favourable_cells', 'unfavourable_cells', 'tie_cells',
        'smallest_advantage_cell_id', 'smallest_advantage_pp',
        'best_applicable_method', 'sort_rank'
    ]
    out = pd.concat([df_a, df_b, df_c], ignore_index=True)[columns]
    return df_a, df_b, df_c, out

def derive():
    input_path = DATA / 'paired_cell_effects.csv'
    output_path = DATA / 'backend_invariance_atlas.csv'
    manifest_path = DATA / 'FIGURE_DATA_DERIVATION_MANIFEST.json'
    coverage_path = DATA / 'table_26_protocol_coverage.csv'
    registry_path = DATA / 'cell_registry_56.csv'

    # Pre-state hash check for input integrity
    inp_sha_before = compute_sha256(input_path)

    inp = pd.read_csv(input_path)

    # 1. Input integrity checks
    if len(inp) != 56:
        raise AssertionError(f"Check failed: len(input) = {len(inp)} != 56")
    if inp.model.nunique() != 7:
        raise AssertionError(f"Check failed: input.model.nunique() = {inp.model.nunique()} != 7")
    if inp.dataset.nunique() != 8:
        raise AssertionError(f"Check failed: input.dataset.nunique() = {inp.dataset.nunique()} != 8")
    if len(inp.drop_duplicates(subset=['model', 'dataset'])) != 56:
        raise AssertionError("Check failed: (model, dataset) pairs are not 56 unique pairs")
    if not (inp['delta_pcg_minus_best_pp'].notna().all() and np.isfinite(inp['delta_pcg_minus_best_pp']).all()):
        raise AssertionError("Check failed: non-finite or NaN in delta_pcg_minus_best_pp")
    if inp.cell_id.nunique() != len(inp):
        raise AssertionError("Check failed: non-unique cell_id values")

    # Verify 40 generations per cell against design contracts
    cov = pd.read_csv(coverage_path).set_index('Quantity')['Value'].astype(int)
    reg = pd.read_csv(registry_path)
    if cov.get('Generations per cell') != 40:
        raise AssertionError(f"Design contract mismatch: table_26 generations != 40 ({cov.get('Generations per cell')})")
    if not (reg['generations'] == 40).all():
        raise AssertionError("Design contract mismatch: cell_registry_56 generations != 40")

    df_a, df_b, df_c, df_all = derive_strata(inp)

    # 2. Stratum construction checks
    if not (df_a['n_cells'] == 8).all():
        raise AssertionError("Check failed: Panel A rows do not all have n_cells == 8")
    if not (df_b['n_cells'] == 7).all():
        raise AssertionError("Check failed: Panel B rows do not all have n_cells == 7")
    if df_a['n_cells'].sum() != 56 or df_b['n_cells'].sum() != 56:
        raise AssertionError("Check failed: stratum cell count sum != 56")
    if set(df_a['stratum_label']) != set(inp.model.unique()):
        raise AssertionError("Check failed: Panel A stratum_label set != distinct models")
    if set(df_b['stratum_label']) != set(inp.dataset.unique()):
        raise AssertionError("Check failed: Panel B stratum_label set != distinct datasets")
    if len(df_c) != 56 or set(df_c['stratum_label']) != set(inp.cell_id.unique()):
        raise AssertionError("Check failed: Panel C stratum_label set != distinct cell_ids")

    # 3. Arithmetic consistency checks
    # Display quantity identity: pcg_risk_advantage_pp == -delta_pcg_minus_best_pp
    np.testing.assert_allclose(
        df_all['pcg_risk_advantage_pp'],
        -df_all['delta_pcg_minus_best_pp'],
        err_msg="pcg_risk_advantage_pp != -delta_pcg_minus_best_pp"
    )

    # Weighted mean equals unweighted mean across cells
    input_adv_mean = (-inp['delta_pcg_minus_best_pp']).mean()
    mean_a_weighted = (df_a['pcg_risk_advantage_pp'] * df_a['n_cells']).sum() / 56.0
    mean_b_weighted = (df_b['pcg_risk_advantage_pp'] * df_b['n_cells']).sum() / 56.0
    if abs(mean_a_weighted - input_adv_mean) > 1e-9:
        raise AssertionError(f"Weighted mean A {mean_a_weighted} != input mean {input_adv_mean}")
    if abs(mean_b_weighted - input_adv_mean) > 1e-9:
        raise AssertionError(f"Weighted mean B {mean_b_weighted} != input mean {input_adv_mean}")

    # Whisker interval bounds
    if not ((df_all['observed_min_pp'] <= df_all['pcg_risk_advantage_pp'] + 1e-12) &
            (df_all['pcg_risk_advantage_pp'] <= df_all['observed_max_pp'] + 1e-12)).all():
        raise AssertionError("Whisker bounds violated: observed_min_pp <= mean <= observed_max_pp")

    # Counts sum to n_cells
    if not (df_all['favourable_cells'] + df_all['unfavourable_cells'] + df_all['tie_cells'] == df_all['n_cells']).all():
        raise AssertionError("Count categories do not sum to n_cells")

    total_fav = int(((-inp['delta_pcg_minus_best_pp']) > 0).sum())
    if df_a['favourable_cells'].sum() != total_fav or df_b['favourable_cells'].sum() != total_fav:
        raise AssertionError("favourable_cells sums do not match input count")

    # Smallest advantage consistency
    for r in df_a.itertuples():
        grp = inp[inp.model == r.stratum_label]
        grp_adv = -grp['delta_pcg_minus_best_pp']
        min_adv = grp_adv.min()
        if not np.isclose(r.smallest_advantage_pp, min_adv):
            raise AssertionError(f"Smallest advantage value mismatch for Model {r.stratum_label}")
        min_cell = grp.loc[grp_adv == min_adv, 'cell_id'].iloc[0]
        if min_cell != r.smallest_advantage_cell_id:
            raise AssertionError(f"Smallest advantage cell mismatch for Model {r.stratum_label}")

    for r in df_b.itertuples():
        grp = inp[inp.dataset == r.stratum_label]
        grp_adv = -grp['delta_pcg_minus_best_pp']
        min_adv = grp_adv.min()
        if not np.isclose(r.smallest_advantage_pp, min_adv):
            raise AssertionError(f"Smallest advantage value mismatch for Dataset {r.stratum_label}")
        min_cell = grp.loc[grp_adv == min_adv, 'cell_id'].iloc[0]
        if min_cell != r.smallest_advantage_cell_id:
            raise AssertionError(f"Smallest advantage cell mismatch for Dataset {r.stratum_label}")

    # Panel C sorting
    if sorted(df_c['sort_rank'].tolist()) != list(range(56)):
        raise AssertionError("Panel C sort_rank is not a permutation of 0..55")
    if not df_c['pcg_risk_advantage_pp'].is_monotonic_increasing:
        raise AssertionError("Panel C pcg_risk_advantage_pp is not monotonically non-decreasing")

    # Panel C rank 0 is global argmin(pcg_risk_advantage_pp)
    global_min_adv = (-inp['delta_pcg_minus_best_pp']).min()
    if not np.isclose(df_c.iloc[0]['pcg_risk_advantage_pp'], global_min_adv):
        raise AssertionError("Panel C rank 0 is not the minimum advantage cell")

    # 4. Provenance checks
    fresh_inp = pd.read_csv(input_path)
    _, _, _, df_all_fresh = derive_strata(fresh_inp)
    pd.testing.assert_frame_equal(df_all, df_all_fresh)

    # Write output CSV
    df_all.to_csv(output_path, index=False, float_format='%.17g', lineterminator='\n', encoding='utf-8')

    # Input immutability assertion
    inp_sha_after = compute_sha256(input_path)
    if inp_sha_before != inp_sha_after:
        raise AssertionError("Input paired_cell_effects.csv was modified during derivation")

    # Idempotence assertion
    bytes_first = output_path.read_bytes()
    _, _, _, df_all_second = derive_strata(pd.read_csv(input_path))
    temp_bytes = df_all_second.to_csv(index=False, float_format='%.17g', lineterminator='\n', encoding='utf-8').encode('utf-8')
    if bytes_first != temp_bytes:
        raise AssertionError("Derivation is not byte-identical on rerun")

    # Update manifest
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('FABRICATED_VALUES') != 0:
        raise AssertionError("FABRICATED_VALUES != 0 before manifest update")

    out_sha = compute_sha256(output_path)
    script_sha = compute_sha256(__file__)
    smallest_c_row = df_c.iloc[0]
    smallest_cell_id = str(smallest_c_row['smallest_advantage_cell_id'])
    best_method = str(smallest_c_row['best_applicable_method'])

    manifest.setdefault('new_csvs', {})['backend_invariance_atlas.csv'] = {
        'source_files': ['paired_cell_effects.csv'],
        'derivation': 'Hierarchical invariance atlas across models (Panel A), datasets (Panel B), and ordered cells (Panel C) using observed extrema whiskers and explicit accepted-risk advantage',
        'input_path': 'data/paired_cell_effects.csv',
        'input_sha256': inp_sha_after,
        'output_path': 'data/backend_invariance_atlas.csv',
        'output_sha256': out_sha,
        'derivation_script_sha256': script_sha,
        'whisker_method': 'OBSERVED_EXTREMA_RANGE',
        'row_counts': '7/8/56',
        'panel_a_rows': 7,
        'panel_b_rows': 8,
        'panel_c_rows': 56,
        'row_count': 71,
        'smallest_advantage_cell_id': smallest_cell_id,
        'smallest_advantage_best_applicable_method': best_method,
        'columns': list(df_all.columns),
        'sha256': out_sha
    }

    if manifest.get('FABRICATED_VALUES') != 0:
        raise AssertionError("FABRICATED_VALUES != 0 after manifest update")

    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f"DERIVATION_SUCCESS=YES ROWS={len(df_all)} OUTPUT={output_path.name} SHA256={out_sha}")
    return df_all

if __name__ == '__main__':
    derive()
