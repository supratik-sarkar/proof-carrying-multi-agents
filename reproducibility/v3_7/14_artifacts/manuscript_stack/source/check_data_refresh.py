"""Check data-contract fidelity and deterministic regeneration without data edits."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figure_contract import ROOT,NAMES,STATIC
from build_figure_architecture import protected,digest
import render_result_figures as r

WORK=ROOT.parent/'.cache/figure-freeze/repro-check'

def main():
    before=protected();source={str(p):digest(p) for p in (ROOT/'source').glob('*') if p.is_file()};images={str(p):digest(p) for p in (ROOT/'images').glob('*') if p.is_file()}
    captured={};original=r.finish
    def capture(f,n):captured[n]=f
    r.finish=capture
    checks=[]
    for n in r.FUNCTIONS:r.generate(n)
    s=pd.read_csv(ROOT/'data/method_summary.csv')
    np.testing.assert_allclose([b.get_width() for b in captured[1].axes[0].patches],s.risk_gap_vs_pcg_pp);checks.append('Figure 1 bars use comparator-minus-PCG reductions')
    np.testing.assert_allclose([b.get_height() for b in captured[1].axes[4].patches],s.mean_token_multiplier)
    np.testing.assert_allclose(captured[1].axes[5].collections[0].get_offsets()[:,1],s.risk_reduction_vs_nocert_pp)
    p=pd.read_csv(ROOT/'data/paired_cell_effects.csv')
    np.testing.assert_allclose(captured[1].axes[1].images[0].get_array(),-p.pivot(index='model',columns='dataset',values='delta_pcg_minus_best_pp'))
    np.testing.assert_allclose(captured[5].axes[0].images[0].get_array(),p.pivot(index='model',columns='dataset',values='pcg_risk'))
    d=pd.read_csv(ROOT/'data/method_dataset_summary.csv')
    for ax,col in [(captured[5].axes[1],'mean_risk'),(captured[5].axes[2],'mean_coverage')]:
        expected=d.pivot(index='method',columns='dataset',values=col).to_numpy();np.testing.assert_allclose(ax.images[0].get_array().filled(np.nan),expected,equal_nan=True)
    checks.append('Figure 1 cost tracks and Figure 5 heatmaps match declared cells and summaries')
    d=pd.read_csv(ROOT/'data/ablation_summary.csv');a=d[d.dataset=='ALL_DATASETS']
    np.testing.assert_allclose([b.get_width() for b in captured[4].axes[0].patches],a.median_risk_increase_pp)
    a=d.pivot(index='variant',columns='dataset',values='median_risk_increase_pp').reindex(d.variant.drop_duplicates());cols=[c for c in a.columns if c!='ALL_DATASETS']+['ALL_DATASETS'];np.testing.assert_allclose(captured[12].axes[0].images[0].get_array(),a[cols]);checks.append('Ablation panels use declared medians without recomputation')
    d=pd.read_csv(ROOT/'data/audit_sampling.csv')
    for ax,col in zip(captured[13].axes[:3],['worst_stratum_coverage','envelope','ess']):np.testing.assert_allclose(ax.collections[1].get_offsets()[:,0],d[col])
    checks.append('Audit scorecard coordinates equal supplied quantities')
    d=pd.read_csv(ROOT/'data/injection_stress.csv');ax=captured[15].axes[0]
    for container,col in zip(ax.containers,['attack_success','accepted_attack','detection']):np.testing.assert_allclose([b.get_height() for b in container],d[col])
    np.testing.assert_allclose(captured[15].axes[1].collections[1].get_offsets()[:,0],d.rho_ucb);checks.append('Injection values and dependence factors come from the same regimes')
    d=pd.read_csv(ROOT/'data/shift_stress.csv')
    for line,(mode,g) in zip(captured[16].axes[0].lines,d.groupby('mode',sort=False)):
        g=g.sort_values('severity');np.testing.assert_allclose(line.get_xdata(),g.severity);np.testing.assert_allclose(line.get_ydata(),g.bound_violation)
    checks.append('Shift trajectories use the supplied severity ordering and values')
    d=pd.read_csv(ROOT/'data/responsibility.csv')
    for collection,col in zip(captured[17].axes[0].collections[1:],['top1','top3']):np.testing.assert_allclose(collection.get_offsets()[:,0],d[col])
    for ax,col in zip(captured[17].axes[1:],['unresolved','median_margin']):np.testing.assert_allclose([b.get_width() for b in ax.patches],d[col])
    checks.append('Responsibility tracks preserve their separate non-additive metrics')
    d=pd.read_csv(ROOT/'data/privacy_frontier.csv').sort_values('privacy_parameter');np.testing.assert_allclose(captured[18].axes[1].lines[0].get_ydata(),d.uncertainty)
    d=pd.read_csv(ROOT/'data/scaling_surface.csv');np.testing.assert_allclose(captured[19].axes[0].images[0].get_array(),d.pivot(index='support_size',columns='redundancy',values='cost_multiplier'));checks.append('Privacy uncertainty and scaling surface equal supplied scenario inputs')
    assert 'no S/V attribution' in ' '.join(t.get_text() for t in captured[4].axes[2].texts);checks.append('Unsupported decomposition is replaced by a descriptive operating-point diagnostic')
    d=pd.read_csv(ROOT/'data/figure_C1_certificate_failure_channel_atlas.csv')
    for ax,panel,order in zip(captured[11].axes,['online_certificate_channels','audit_sidecar'],[['V_H','V_Pi','V_Gamma','V_entail'],['ReplayFail','DriftFail','CovGap']]):np.testing.assert_allclose([b.get_width() for b in ax.patches],d[d.panel==panel].set_index('channel').loc[order,'value'])
    checks.append('Figure 11 uses supplied channel metrics without reinterpretation')
    d=pd.read_csv(ROOT/'data/witness_matrix_scoped.csv');a=d.pivot(index='method',columns='witness',values='plot_value').reindex(d.method.drop_duplicates());actual=captured[14].axes[0].images[0].get_array()
    np.testing.assert_array_equal(np.ma.getmaskarray(actual),a.isna());np.testing.assert_allclose(actual.filled(np.nan),a.to_numpy(),equal_nan=True);checks.append('Native N/A mask and witness values match exactly')
    d=pd.read_csv(ROOT/'data/timing_summary.csv');ax=captured[20].axes[0]
    for i,row in enumerate(d.itertuples()):
        np.testing.assert_allclose(ax.lines[i*2].get_xdata(),[row.median_latency_s,row.p95_latency_s]);np.testing.assert_allclose(ax.lines[i*2+1].get_xdata(),[row.p95_latency_s])
    checks.append('Latency markers use supplied median and p95')
    d=pd.read_csv(ROOT/'data/figure_D2_cost_telemetry.csv')
    p=d[d.row_type=='PCG_PHASE']
    np.testing.assert_allclose([b.get_width() for b in captured[21].axes[0].patches],p.absolute_normalized_cost)
    np.testing.assert_allclose([b.get_width() for b in captured[21].axes[1].patches],100*p.share)
    checks.append('PCG-only contributions and path shares equal declared cost accounting')
    for f in captured.values():plt.close(f)
    r.finish=original;WORK.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PCG_MANUSCRIPT_IMAGES=str(WORK),PYTHONDONTWRITEBYTECODE='1')
    for script in ['render_result_figures.py','render_static_schematics.py']:subprocess.run([sys.executable,str(ROOT/'source'/script)],env=env,check=True)
    for name in NAMES:
        for ext in ['.pdf','.png']:assert digest(WORK/(name+ext))==digest(ROOT/'images'/(name+ext)),name+ext
    assert protected()==before
    assert all(digest(Path(p))==h for p,h in source.items())
    assert all(digest(Path(p))==h for p,h in images.items())
    report={'status':'PASS','checks':checks,'deterministic_pdf_png_pairs':len(NAMES),'data_files_modified':0,'table_files_modified':0,'source_files_modified':0,'production_images_modified':0,'scope':'Data-contract fidelity and same-input reproducibility'}
    (ROOT.parent/'experiment_provenance/data_refresh_check.json').write_text(json.dumps(report,indent=2)+'\n');print('DATA_CONTRACT_FIDELITY=PASS DETERMINISTIC_FIGURES=23/23 PROTECTED_FILES_UNCHANGED=PASS')
if __name__=='__main__':main()
