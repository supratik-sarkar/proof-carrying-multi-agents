"""Stable display conventions and data access shared by result-driven figures."""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('PCG_MANUSCRIPT_DATA', ROOT/'data'))
OUT = Path(os.environ.get('PCG_MANUSCRIPT_IMAGES', ROOT/'images'))
OUT.mkdir(parents=True, exist_ok=True)
REGISTRY = pd.read_csv(DATA/'cell_registry_56.csv')
MODELS = REGISTRY.model.drop_duplicates().tolist()
DATASETS = REGISTRY.dataset.drop_duplicates().tolist()
METHODS = ['NoCert','MiniCheck','AlignScore','QAFactEval','CMVO','SignalMatchedFusion','PCG-MAS']
PARAMETERS = json.loads((DATA/'plot_parameters.json').read_text())

def metrics():
    df = pd.read_csv(DATA/'cell_metrics.csv')
    df['applicable'] = df['applicable'].astype(str).str.lower().map({'true': True, 'false': False})
    if df.applicable.isna().any(): raise ValueError('applicable must be true or false')
    for key in ['risk','coverage','token_multiplier']:
        if not np.isfinite(df.loc[df.applicable, key]).all():
            raise ValueError(f'Nonfinite applicable metric: {key}')
    return df

def token_multipliers(df):
    # Use the mean across applicable cells if the new run has cell-dependent cost.
    return df[df.applicable].groupby('method').token_multiplier.mean().to_dict()

def panel_description():
    values = REGISTRY.generations.unique()
    count = f'{values[0]} generations per cell' if len(values) == 1 else 'generation counts in registry'
    return f'{len(MODELS)} models × {len(DATASETS)} datasets × {count}'

def contrast(image, value):
    if not np.isfinite(value): return '#17202A'
    rgb = np.asarray(image.cmap(image.norm(value)))[:3]
    return 'white' if np.dot(rgb, [0.2126,0.7152,0.0722]) < .48 else '#17202A'

def effect_limit(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return max(float(np.max(np.abs(finite))) if finite.size else 0, .01)

def point_labels(ax, points):
    """Place point labels deterministically, testing their rendered bounds."""
    ax.margins(x=.18, y=.22)
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    offsets = [(8,8), (8,-14), (-8,8), (-8,-14), (8,24), (8,-30), (-8,24), (-8,-30)]
    for x, y, label in points:
        annotation = ax.annotate(label, (x,y), xytext=(8,8), textcoords='offset points', fontsize=7,
                                 arrowprops={'arrowstyle':'-', 'color':'#78909C', 'lw':.5})
        for dx,dy in offsets:
            annotation.set_position((dx,dy))
            annotation.set_ha('left' if dx>0 else 'right')
            bounds = annotation.get_window_extent(renderer).expanded(1.05,1.1)
            if ax.bbox.contains(bounds.x0,bounds.y0) and ax.bbox.contains(bounds.x1,bounds.y1) and not any(bounds.overlaps(old) for old in placed):
                break
        placed.append(bounds)

def save(fig, name):
    fig.savefig(OUT/(name+'.pdf'), bbox_inches='tight', metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(OUT/(name+'.png'), dpi=240, bbox_inches='tight')
    plt.close(fig)
