"""Render Figure 06: Backend Invariance Atlas.

Reads strictly from data/backend_invariance_atlas.csv and renders
the three-panel hierarchical invariance atlas to PDF and PNG.
Adheres to descriptive observed extrema summary and explicit accepted-risk advantage.
"""
from pathlib import Path
import hashlib, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('PCG_MANUSCRIPT_DATA', ROOT / 'data'))
OUT = Path(os.environ.get('PCG_MANUSCRIPT_IMAGES', ROOT / 'images'))
OUT.mkdir(parents=True, exist_ok=True)

INK = '#183044'
BLUE = '#2374AB'
TEAL = '#008978'
GRAY = '#8495A4'

def compute_sha256(filepath):
    return hashlib.sha256(Path(filepath).read_bytes()).hexdigest()

def title(ax, letter, head, sub=''):
    ax.set_title(letter + '  ' + head + ('\n' + sub if sub else ''), loc='left', fontweight='bold', pad=10, fontsize=9.5, linespacing=1.3)

def tidy(ax, axis='x'):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color='#E8EDF0', lw=0.7)

def render():
    input_file = DATA / 'backend_invariance_atlas.csv'
    manifest_file = DATA / 'FIGURE_DATA_DERIVATION_MANIFEST.json'
    coverage_file = DATA / 'table_26_protocol_coverage.csv'
    registry_file = DATA / 'cell_registry_56.csv'

    if not input_file.exists():
        raise FileNotFoundError(f"Missing authoritative input: {input_file}")
    if not manifest_file.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_file}")
    if not coverage_file.exists():
        raise FileNotFoundError(f"Missing coverage file: {coverage_file}")
    if not registry_file.exists():
        raise FileNotFoundError(f"Missing registry file: {registry_file}")

    # Integrity gate: Assert SHA-256 matches manifest
    actual_sha = compute_sha256(input_file)
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    entry = manifest.get('new_csvs', {}).get('backend_invariance_atlas.csv', {})
    expected_sha = entry.get('sha256') or entry.get('output_sha256')

    if actual_sha != expected_sha:
        raise ValueError(f"Integrity check failed: backend_invariance_atlas.csv SHA {actual_sha} != manifest SHA {expected_sha}")

    # Design contract verification: 40 generations/cell
    cov = pd.read_csv(coverage_file).set_index('Quantity')['Value'].astype(int)
    reg = pd.read_csv(registry_file)
    if cov.get('Generations per cell') != 40 or not (reg['generations'] == 40).all():
        raise ValueError("Design contract mismatch: generations per cell != 40")

    # Read ONLY data/backend_invariance_atlas.csv
    df = pd.read_csv(input_file)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#B8C4CB',
        'text.color': INK,
        'axes.labelcolor': INK,
        'pdf.fonttype': 42,
        'savefig.facecolor': 'white'
    })

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3), gridspec_kw={'width_ratios': [1.0, 1.0, 1.35], 'wspace': 0.42})
    fig.subplots_adjust(left=0.13, right=0.985, top=0.82, bottom=0.18)

    # ----------------------------------------------------
    # Panel A — per-model invariance
    # ----------------------------------------------------
    df_a = df[df.panel == 'A'].sort_values('pcg_risk_advantage_pp', ascending=True).reset_index(drop=True)
    y_a = np.arange(len(df_a))
    xerr_a = [
        df_a['pcg_risk_advantage_pp'] - df_a['observed_min_pp'],
        df_a['observed_max_pp'] - df_a['pcg_risk_advantage_pp']
    ]

    axs[0].errorbar(
        df_a['pcg_risk_advantage_pp'], y_a, xerr=xerr_a,
        fmt='o', color=TEAL, ecolor=TEAL, elinewidth=1.4, capsize=2, capthick=1.4, markersize=5
    )
    axs[0].axvline(0, color=INK, lw=1.2)
    axs[0].set_yticks(y_a)
    axs[0].set_yticklabels(df_a['stratum_label'])
    axs[0].invert_yaxis()
    axs[0].set_xlabel('Accepted-risk advantage (pp)')
    title(axs[0], 'A', 'Per-model invariance', '8 datasets/model · 40 generations/cell')
    tidy(axs[0], axis='x')

    # ----------------------------------------------------
    # Panel B — per-dataset invariance
    # ----------------------------------------------------
    df_b = df[df.panel == 'B'].sort_values('pcg_risk_advantage_pp', ascending=True).reset_index(drop=True)
    y_b = np.arange(len(df_b))
    xerr_b = [
        df_b['pcg_risk_advantage_pp'] - df_b['observed_min_pp'],
        df_b['observed_max_pp'] - df_b['pcg_risk_advantage_pp']
    ]

    axs[1].errorbar(
        df_b['pcg_risk_advantage_pp'], y_b, xerr=xerr_b,
        fmt='o', color=BLUE, ecolor=BLUE, elinewidth=1.4, capsize=2, capthick=1.4, markersize=5
    )
    axs[1].axvline(0, color=INK, lw=1.2)
    axs[1].set_yticks(y_b)
    axs[1].set_yticklabels(df_b['stratum_label'])
    axs[1].invert_yaxis()
    axs[1].set_xlabel('Accepted-risk advantage (pp)')
    title(axs[1], 'B', 'Per-dataset invariance', '7 models/dataset · 40 generations/cell')
    tidy(axs[1], axis='x')

    # Joint x-limits for Panels A & B
    joint_min = min(df_a['observed_min_pp'].min(), df_b['observed_min_pp'].min())
    joint_max = max(df_a['observed_max_pp'].max(), df_b['observed_max_pp'].max())
    xlim_left = -0.3
    xlim_right = 4.0
    axs[0].set_xlim(xlim_left, xlim_right)
    axs[1].set_xlim(xlim_left, xlim_right)

    # ----------------------------------------------------
    # Panel C — per-cell ordered series
    # ----------------------------------------------------
    df_c = df[df.panel == 'C'].sort_values('sort_rank', ascending=True).reset_index(drop=True)
    x_c = df_c['sort_rank'].to_numpy()
    y_c = df_c['pcg_risk_advantage_pp'].to_numpy()

    axs[2].scatter(x_c, y_c, marker='o', s=16, color=TEAL, alpha=0.85, zorder=3)
    axs[2].axhline(0, color=INK, lw=1.2, zorder=2)

    y_max = 4.0
    y_min = -0.3
    axs[2].set_ylim(y_min, y_max)
    axs[2].set_xlim(-2, 58)
    axs[2].axhspan(0, y_max, color=GRAY, alpha=0.08, zorder=1)

    # Smallest advantage annotation (Rank 0)
    smallest_row = df_c.iloc[0]
    smallest_id = str(smallest_row['smallest_advantage_cell_id'])
    best_method = str(smallest_row['best_applicable_method'])
    annotation_text = f'Smallest advantage: {smallest_id}\nbest comparator: {best_method}'

    axs[2].annotate(
        annotation_text, xy=(0, y_c[0]), xytext=(3, 2.3),
        ha='left', va='bottom', fontsize=7.5, color=INK,
        arrowprops=dict(arrowstyle='->', color=INK, lw=0.7,
                        connectionstyle='arc3,rad=0.38', shrinkA=2, shrinkB=4)
    )

    axs[2].set_xlabel('Model × dataset cells, ordered by PCG-MAS advantage')
    axs[2].set_ylabel('Accepted-risk advantage (pp)')
    axs[2].set_xticks([])

    fav_total = int((y_c > 0).sum())
    total_cells = len(df_c)
    text_block = f'favourable {fav_total}/{total_cells} cells\n40 generations/cell'
    axs[2].text(
        0.04, 0.92, text_block, transform=axs[2].transAxes,
        va='top', ha='left', fontsize=8, color=INK, linespacing=1.4
    )
    title(axs[2], 'C', 'Cell-level invariance', '56 model × dataset cells · 40 generations/cell')
    tidy(axs[2], axis='y')

    # Shared convention line centered beneath panels
    fig.text(0.5, 0.03, r'Advantage = $R_{\rm best\ applicable} - R_{\rm PCG-MAS}$ (pp); positive favors PCG-MAS',
             ha='center', va='bottom', fontsize=8, color=INK)

    # Save PDF and PNG
    pdf_out = OUT / 'figure_06_backend_invariance_atlas.pdf'
    png_out = OUT / 'figure_06_backend_invariance_atlas.png'
    fig.savefig(pdf_out, metadata={'CreationDate': None, 'ModDate': None}, bbox_inches='tight')
    fig.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"RENDER_SUCCESS=YES PDF={pdf_out.name} PNG={png_out.name}")

if __name__ == '__main__':
    render()
