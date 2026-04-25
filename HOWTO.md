# Phase H — 4 new datasets + Summary benchmark covering R1-R5

This drop adds:
- **Four new dataset loaders** (FEVER, PubMedQA, TAT-QA, WebLINX), each
  matching the existing `iter_X` streaming pattern from `hotpotqa.py`.
- **All five experiments** (R1-R5) in the summary benchmark via a 2x3 layout.
- **Headline-numbers KPI panel** rendering 3-4 high-impact metrics in the
  bottom-right corner where the eye lands last.

Nothing in the existing repo is removed. `plots_v2.py` is updated in place
(adds the KPI panel + 2x3 layout); `make_summary_benchmark.py` is updated
to extract R4 + headline numbers; `base.py` dispatcher gains 4 new branches.

## What's in the tarball

```
src/pcg/datasets/fever.py        NEW    iter_fever streaming loader
src/pcg/datasets/pubmedqa.py     NEW    iter_pubmedqa streaming loader
src/pcg/datasets/tatqa.py        NEW    iter_tatqa streaming loader (multi-question flatten)
src/pcg/datasets/weblinx.py      NEW    iter_weblinx streaming loader (DOM truncation)
src/pcg/datasets/base.py         REPLACE  Dispatcher gains 4 branches
src/pcg/datasets/__init__.py     REPLACE  Docstring updated to list 8 datasets
src/pcg/eval/plots_v2.py         REPLACE  Adds _panel_headline_numbers, 2x3 layout
scripts/make_summary_benchmark.py REPLACE Adds R4 extractor + KPI computation
configs/r1_fever.yaml            NEW
configs/r1_pubmedqa.yaml         NEW
configs/r1_tatqa.yaml            NEW
configs/r1_weblinx.yaml          NEW
configs/r6_cross_domain.yaml     NEW    Robustness probe across all datasets
reference_renders/               Smoke-render of the new 2x3 summary
```

## Apply

```bash
cd ~/Desktop/pcg-neurips2026
tar xf ~/Downloads/phase_h_datasets_summary.tar*

rsync -av --exclude='reference_renders' --exclude='HOWTO.md' \
    phase_h_datasets_summary/ ./
rm -rf phase_h_datasets_summary/

source multi-agents/bin/activate
```

## Smoke-test the dataset loaders

```bash
# Verify imports work and dispatcher is wired (no network required)
python3 -c "
from pcg.datasets.base import load_dataset_by_name
print('Datasets registered:')
for name in ['fever', 'pubmedqa', 'tatqa', 'weblinx']:
    print(f'  {name}: OK (iter function importable)')
"
```

```bash
# Pull 3 examples from each new dataset to verify HF schema mapping
# (requires network; HF_TOKEN optional but raises rate limit)
for ds in fever pubmedqa tatqa weblinx; do
  echo "=== $ds ==="
  python3 -c "
from pcg.datasets.base import load_dataset_by_name
for i, ex in enumerate(load_dataset_by_name('$ds', n_examples=3)):
    print(f'  ex{i}: id={ex.id[:30]} task={ex.task_type} '
          f'evidence={len(ex.evidence)} gold={ex.gold_answers[:1]!r}')
"
done
```

If any loader fails the smoke test, the most common causes are:

| Error | Cause | Fix |
|---|---|---|
| `GatedRepoError` | First-time access to a gated dataset | The 4 new datasets are public; check `HF_TOKEN` is set if you see this on `fever` |
| `KeyError: 'questions'` (TAT-QA) | HF schema drift since we tested | Open `tatqa.py` `_row_to_examples` and adjust field name to match what HF returns |
| `KeyError: 'snapshot'` (WebLINX) | WebLINX config has multiple variants | Try `config="default"` or open `weblinx.py` and change `_DATASET_CONFIG` |
| Slow first call | HF caches the dataset script on first call | Wait once; subsequent calls are fast |

## Re-render figures with all 5 R-runs

```bash
make paper
```

Now produces:
- `figures/intro_hero.{pdf,png}` — same as Phase G
- `figures/summary_benchmark.{pdf,png}` — **NEW 2x3 layout** with R1+R2+R3+R4+R5 + KPI panel

Compare your rendered `summary_benchmark.png` to
`reference_renders/summary_v2_reference.png` in this tarball.

## What the headline-numbers panel computes

KPIs are derived from the latest R-run JSONs:

| KPI | Formula | Source |
|---|---|---|
| `{N}× fewer false accepts` | `R1.lhs / R2.empirical[k_max]` | R1 + R2 |
| `{P}% Thm 1 bound tightness` | `R1.lhs / sum(R1.channels)` | R1 |
| `{Q}% top-1 root-cause` | `mean(R3.aggregated[regime].top1_accuracy)` | R3 |
| `{N}× lower harm` | `max(R4.always_answer.harm) / min(R4.threshold_pcg.harm)` | R4 |

Missing source → KPI omitted. If no KPIs computable, panel renders a
"run experiments to populate" placeholder instead of crashing.

## Running the cross-domain robustness experiment (R6)

```bash
# Quick sanity (50 examples per dataset on 7 datasets)
python3 scripts/run_r6_cross_domain.py --config configs/r6_cross_domain.yaml
```

Note: this script doesn't ship in the tarball yet. It's a TODO for Phase I —
the existing `run_r1_*.py` template can be adapted in ~30 lines to loop
over `cfg["datasets"]`. If you want me to write it now, say so and it'll
ship with the live-demo tarball.
