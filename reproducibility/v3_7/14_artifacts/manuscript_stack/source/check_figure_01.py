"""Check Figure 1 display signs and values without building other artifacts."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import render_result_figures as renderer
from figure_contract import ROOT

def main():
    captured=[]; renderer.finish=lambda figure,number:captured.append(figure)
    renderer.generate(1);f=captured[0];a,b,c,d,e,g,scale=f.axes
    s=renderer.read('method_summary.csv');p=renderer.read('paired_cell_effects.csv')
    np.testing.assert_allclose([bar.get_width() for bar in a.patches],s.risk_gap_vs_pcg_pp)
    np.testing.assert_allclose(b.images[0].get_array(),-p.pivot(index='model',columns='dataset',values='delta_pcg_minus_best_pp'))
    np.testing.assert_allclose([c.collections[i*3+1].get_offsets()[0].tolist() for i in range(len(s))],s[['mean_coverage','mean_risk']])
    assert c.get_legend() is None and d.get_legend() is None
    for i,dataset in enumerate(p.dataset.drop_duplicates()):
        values=-p.loc[p.dataset==dataset,'delta_pcg_minus_best_pp'].to_numpy()
        np.testing.assert_allclose(d.collections[i*2].get_offsets()[:,0],values)
        np.testing.assert_allclose(d.collections[i*2+1].get_offsets()[0,0],np.median(values))
        np.testing.assert_allclose(d.lines[1+i*2].get_xdata(),[values.min(),values.max()])
        np.testing.assert_allclose(d.lines[2+i*2].get_xdata(),np.quantile(values,[.25,.75],method='linear'))
    assert len([t for t in c.texts if t.get_text()])==len(s)
    assert len([t for t in d.texts if t.get_text() and 'Median' not in t.get_text()])==p.dataset.nunique()
    np.testing.assert_allclose([bar.get_height() for bar in e.patches],s.mean_token_multiplier)
    np.testing.assert_allclose(g.collections[0].get_offsets()[:,1],s.risk_reduction_vs_nocert_pp)
    assert 2 in e.get_yticks()
    assert a.get_xlim()[0]==0 and d.get_xlim()[0]==0
    assert np.isclose(s.loc[s.method=='PCG-MAS','risk_gap_vs_pcg_pp'].iloc[0],0)
    plt.close(f)
    print('PASS: all six Figure 1 tracks equal stored inputs under explicit display sign conventions.')
if __name__=='__main__':main()
