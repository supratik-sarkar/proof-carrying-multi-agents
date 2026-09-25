"""Validate the complete stored asset set and Cartesian metric registry."""
from pathlib import Path
import csv
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
METHODS = {'NoCert', 'MiniCheck', 'AlignScore', 'QAFactEval', 'CMVO', 'SignalMatchedFusion', 'PCG-MAS'}

def validate(root=ROOT, inputs_only=False):
    errors = []
    counts = {}
    for name, folder, pattern, expected in [
        ('FIGURE_PDF', 'images', '*.pdf', 23), ('FIGURE_PNG', 'images', '*.png', 23),
        ('TABLE_CSV', 'data', 'table_*.csv', 36), ('TABLE_TEX', 'tables', 'table_*.tex', 34),
    ]:
        if inputs_only and folder != 'data': continue
        paths = list((root / folder).glob(pattern))
        counts[name] = len(paths)
        if len(paths) != expected:
            errors.append(f'{name}: expected {expected}, found {len(paths)}')
        if any(p.stat().st_size == 0 for p in paths):
            errors.append(f'{name}: empty asset')
    for folder, a, b in ([] if inputs_only else [('images', '*.pdf', '*.png')]):
        if {p.stem for p in (root/folder).glob(a)} != {p.stem for p in (root/folder).glob(b)}:
            errors.append('figure PDF/PNG identities differ')
    if not inputs_only:
        import re
        tex=(root/'main.tex').read_text()
        for name in re.findall(r'\\pcgtable\{([^}]+)\}',tex):
            if not (root/'tables'/(name+'.tex')).is_file():errors.append('Missing table: '+name)
    try:
        with (root/'data/cell_registry_56.csv').open() as f: registry = list(csv.DictReader(f))
        with (root/'data/cell_metrics.csv').open() as f: metrics = list(csv.DictReader(f))
    except (OSError, csv.Error) as e:
        return counts, errors + [str(e)]
    cells = [(r['model'], r['dataset']) for r in registry]
    counts.update(CELL_REGISTRY_ROWS=len(registry), CELL_METRICS_ROWS=len(metrics))
    if len(cells) != 56 or len(set(cells)) != 56: errors.append('registry must contain 56 unique cells')
    if any(not r['generations'].isdigit() or int(r['generations']) <= 0 for r in registry): errors.append('registry generations must be positive integers')
    keys = [(r['model'], r['dataset'], r['method']) for r in metrics]
    expected = {(m, d, method) for m, d in cells for method in METHODS}
    if len(keys) != 392 or set(keys) != expected or any(n != 1 for n in Counter(keys).values()):
        errors.append('metrics must contain exactly one row for each of 56 cells x 7 methods')
    return counts, errors

if __name__ == '__main__':
    counts, errors = validate(inputs_only="--inputs-only" in sys.argv)
    print('PACKAGE_VALIDATION=' + ('FAIL' if errors else 'PASS'))
    for key, value in counts.items(): print(f'{key}={value}')
    for error in errors: print('BLOCKER=' + error)
    sys.exit(2 if errors else 0)
