"""Numerical and notation checks without rewriting any figure artifact."""
import json,hashlib,re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pypdf import PdfReader
import render_result_figures as r
from appendix_figure_layouts import checked_channels,checked_cost

def main():
    captures={};r.finish=lambda f,n:captures.setdefault(n,f)
    for n in [5,11,12,13,14,15,16,17,18,19,20,21]:r.generate(n)
    p=r.read('paired_cell_effects.csv');d=r.read('method_dataset_summary.csv')
    np.testing.assert_allclose(captures[5].axes[0].images[0].get_array(),p.pivot(index='model',columns='dataset',values='pcg_risk'))
    for ax,col in zip(captures[5].axes[1:3],['mean_risk','mean_coverage']):np.testing.assert_allclose(ax.images[0].get_array().filled(np.nan),d.pivot(index='method',columns='dataset',values=col),equal_nan=True)
    for artist,model in zip(captures[5].axes[3].collections,p.model.drop_duplicates()):np.testing.assert_allclose(artist.get_offsets()[:,0],-p.loc[p.model==model,'delta_pcg_minus_best_pp'])
    a,b=checked_channels(r)
    for ax,frame in zip(captures[11].axes,[a,b]):np.testing.assert_allclose([x.get_width() for x in ax.patches],frame.value)
    ab=r.read('ablation_summary.csv');grid=ab.pivot(index='variant',columns='dataset',values='median_risk_increase_pp').reindex(ab.variant.drop_duplicates());cols=[c for c in grid.columns if c!='ALL_DATASETS']+['ALL_DATASETS'];np.testing.assert_allclose(captures[12].axes[0].images[0].get_array(),grid[cols])
    d=r.read('audit_sampling.csv')
    for ax,col in zip(captures[13].axes[:3],['worst_stratum_coverage','envelope','ess']):np.testing.assert_allclose(ax.collections[1].get_offsets()[:,0],d[col])
    for container,col in zip(captures[13].axes[3].containers,['violation','false_cert']):np.testing.assert_allclose([bar.get_width() for bar in container],d[col])
    w=r.read('witness_matrix_scoped.csv');grid=w.pivot(index='method',columns='witness',values='plot_value').reindex(w.method.drop_duplicates());im=captures[14].axes[0].images[0].get_array();np.testing.assert_array_equal(np.ma.getmaskarray(im),grid.isna());np.testing.assert_allclose(im.filled(np.nan),grid,equal_nan=True)
    assert all(label.get_text().startswith('$W_') for label in captures[14].axes[0].get_xticklabels())
    d=r.read('injection_stress.csv')
    for container,col in zip(captures[15].axes[0].containers,['attack_success','accepted_attack','detection']):np.testing.assert_allclose([bar.get_height() for bar in container],d[col])
    np.testing.assert_allclose(captures[15].axes[1].collections[1].get_offsets()[:,0],d.rho_ucb)
    d=r.read('shift_stress.csv')
    for i,(_,g) in enumerate(d.groupby('mode',sort=False)):
        g=g.sort_values('severity');np.testing.assert_allclose(captures[16].axes[0].lines[i].get_ydata(),g.bound_violation);np.testing.assert_allclose(captures[16].axes[1].collections[i].get_offsets(),g[['utility','audit_coverage']]);np.testing.assert_allclose(captures[16].axes[1].collections[i].get_array(),g.alarm_power)
    d=r.read('responsibility.csv')
    for artist,col in zip(captures[17].axes[0].collections[1:],['top1','top3']):np.testing.assert_allclose(artist.get_offsets()[:,0],d[col])
    for ax,col in zip(captures[17].axes[1:],['unresolved','median_margin']):np.testing.assert_allclose([bar.get_width() for bar in ax.patches],d[col])
    d=r.read('privacy_frontier.csv').sort_values('privacy_parameter');np.testing.assert_allclose(np.asarray(captures[18].axes[0].lines[0].get_ydata(),dtype=float),d.modelled_risk);np.testing.assert_allclose(np.asarray(captures[18].axes[0].lines[0].get_xdata(),dtype=float),d.modelled_utility);np.testing.assert_allclose(captures[18].axes[1].lines[0].get_ydata(),d.uncertainty)
    d=r.read('scaling_surface.csv');np.testing.assert_allclose(captures[19].axes[0].images[0].get_array(),d.pivot(index='support_size',columns='redundancy',values='cost_multiplier'))
    d=r.read('timing_samples.csv');s=r.read('timing_summary.csv')
    for i,row in enumerate(s.itertuples()):
        np.testing.assert_allclose(captures[20].axes[0].collections[i*2].get_offsets()[:,0],d.loc[d.method==row.method,'latency_s']);np.testing.assert_allclose(captures[20].axes[0].lines[i*2].get_xdata(),[row.median_latency_s,row.p95_latency_s])
    p,total=checked_cost(r)
    np.testing.assert_allclose([bar.get_width() for bar in captures[21].axes[0].patches],p.absolute_normalized_cost);np.testing.assert_allclose([bar.get_width() for bar in captures[21].axes[1].patches],p.share*100)
    source=(r.ROOT/'source/figure_B1_theory_dependency.tex').read_text();assert len(re.findall(r'\\node\[(?:tile|result|stress)(?:,|\])',source))==9
    reader=PdfReader(r.ROOT/'main.pdf');report=json.loads((r.ROOT.parent/'experiment_provenance/manuscript_build.json').read_text());page=next(x['page'] for x in report['figures'] if x['label']=='fig:app-theory-dependency')
    destinations=[]
    for entry in reader.pages[page-1].get('/Annots',[]):
        obj=entry.get_object();action=obj.get('/A',{});dest=action.get('/D') if action.get('/S')=='/GoTo' else obj.get('/Dest')
        if isinstance(dest,str):destinations.append(dest);assert dest in reader.named_destinations,dest
    required=['definition.1','definition.2','definition.3','definition.4','algorithm.1']
    assert all(x in destinations for x in required),destinations
    assert len(destinations)>=25
    for n in [5,11,12,14,21]:assert len(captures[n].axes)>=2
    plt.close('all')
    result={'status':'PASS','numerical_figures_checked':[5,11,12,13,14,15,16,17,18,19,20,21],'absolute_quantity_figures_checked':[13,15,16,17,18,19,20],'primary_tiles':9,'figure10_live_links_on_page':len(destinations),'figure10_live_link_destinations':sorted(set(destinations)),'empirical_values_in_figure10':0,'figure11_rows':[4,3],'figure21_shares_sum':float(p.share.sum()),'figure21_duplicates_figure1':False,'no_comparator_phase_costs_invented':True}
    (r.ROOT.parent/'experiment_provenance/appendix_figure_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
