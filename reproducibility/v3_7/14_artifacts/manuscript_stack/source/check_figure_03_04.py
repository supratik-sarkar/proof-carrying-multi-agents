"""Narrow data-contract checks for Figures 3 and 4."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection,PolyCollection
import render_result_figures as r
from figure_03_04_layouts import DATASET_ORDER,render_decomposition,wilson_interval

METHODS=['NoCert','MiniCheck','AlignScore','QAFactEval','CMVO','SignalMatchedFusion','PCG-MAS']

def main():
    captured={};r.finish=lambda figure,number:captured.setdefault(number,figure)
    r.generate(3);r.generate(4)

    # Figure 4B: every aggregate operating point comes from method_summary.csv.
    summary=r.read('method_summary.csv').set_index('method').loc[METHODS]
    b=captured[4].axes[1]
    point_layers=[x for x in b.collections if isinstance(x,PathCollection)]
    assert len(point_layers)==len(METHODS)
    actual=np.array([x.get_offsets()[0] for x in point_layers],dtype=float)
    expected=summary[['mean_coverage','mean_risk']].to_numpy(float)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)

    # Figure 4C: source precision satisfies Delta=S+V and Table 25 matches rounding.
    sv=r.read('selectivity_verification.csv').set_index('dataset').loc[DATASET_ORDER]
    table=r.read('table_25_protocol_sv.csv').set_index('Dataset').loc[DATASET_ORDER]
    np.testing.assert_allclose(sv.selection_pp+sv.verification_pp,sv.total_pp,rtol=0,atol=1e-12)
    np.testing.assert_allclose(sv.selection_pp.round(2),table['S pp'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(sv.verification_pp.round(2),table['V pp'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(sv.total_pp.round(2),table['Delta pp'],rtol=0,atol=1e-12)
    np.testing.assert_array_equal(np.rint(100*sv.verification_share).astype(int),table['Verification share'].str.rstrip('%').astype(int))
    c=captured[4].axes[2]
    n=len(DATASET_ORDER)
    np.testing.assert_allclose([x.get_height() for x in c.patches[:n]],sv.selection_pp)
    np.testing.assert_allclose([x.get_height() for x in c.patches[n:2*n]],sv.verification_pp)
    np.testing.assert_allclose([x.get_y() for x in c.patches[n:2*n]],sv.selection_pp)
    protocol=json.loads((r.DATA/'figure_04_protocol.json').read_text())
    fig,ax=plt.subplots();render_decomposition(ax,sv.reset_index(),table.reset_index(),protocol,r);plt.close(fig)

    # Figure 3C: center lines and shaded 95% Wilson intervals use plot_parameters.json.
    cfg=json.loads((r.DATA/'plot_parameters.json').read_text());au=cfg['audit_theory']
    panel=captured[3].axes[2]
    bands=[x for x in panel.collections if isinstance(x,PolyCollection)]
    assert len(bands)==len(au['uncovered_percent'])
    assert au['confidence_level']==0.95 and 'Wilson' in au['confidence_method']
    for i,(line,u) in enumerate(zip(panel.lines,au['uncovered_percent'])):
        probes=line.get_xdata();sampling=np.sqrt(np.log(au['inverse_delta'])/(2*probes))
        np.testing.assert_allclose(line.get_ydata(),100*(au['residual']+sampling)+u)
        low,high=wilson_interval(au['residual'],probes,au['confidence_level'])
        vertices=bands[i].get_paths()[0].vertices
        y=vertices[:,1]
        assert y.min()<=np.min(100*(low+sampling)+u)+1e-10
        assert y.max()>=np.max(100*(high+sampling)+u)-1e-10
    annotation=' '.join(t.get_text() for t in captured[3].texts)
    assert '95% Wilson' in annotation and 'not confidence intervals' not in annotation

    plt.close('all')
    print('PASS: Figure 3C confidence intervals, Figure 4B operating points, and Figure 4C S/V identity match manuscript data.')

if __name__=='__main__':main()
