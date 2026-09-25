"""Deterministic main-table projections with row-level source attribution."""
from pathlib import Path
import json,os
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('PCG_MANUSCRIPT_DATA',ROOT/'data'))
OUT=Path(os.environ.get('PCG_MANUSCRIPT_TABLES',ROOT/'tables'))
RENAMES={'table_01_main_six_summary':'table_01_primary_endpoint_by_dataset',
 'table_08_baseline_scope_matrix':'table_02_comparator_scope_applicability',
 'table_28_protocol_cost':'table_04_cost_anatomy',
 'table_03_audit_calibration_summary':'table_05_audit_calibration_summary'}

def prepare():
    p=pd.read_csv(DATA/'paired_cell_effects.csv');s=pd.read_csv(DATA/'method_dataset_summary.csv')
    if len(p)!=56 or not p.groupby('dataset').size().eq(7).all():raise ValueError('Incomplete primary-endpoint panel')
    np.testing.assert_allclose(p.pcg_risk-p.best_applicable_risk,p.delta_pcg_minus_best_pp)
    groups=p.groupby('dataset',sort=False)
    t=groups.agg(pcg_accepted_risk=('pcg_risk','mean'),best_applicable_risk=('best_applicable_risk','mean'),delta_primary_pp=('delta_pcg_minus_best_pp','mean'),retained_coverage=('pcg_coverage','mean'),model_cells=('model','size')).reset_index()
    for row in t.itertuples():
        source=s[(s.method=='PCG-MAS')&(s.dataset==row.dataset)].iloc[0]
        np.testing.assert_allclose([row.pcg_accepted_risk,row.retained_coverage],[source.mean_risk,source.mean_coverage])
    t['source_file']='paired_cell_effects.csv';t['aggregation']='Arithmetic mean over the seven paired model cells; comparator chosen within each cell'
    t.to_csv(DATA/'table_01_primary_endpoint_by_dataset.csv',index=False,float_format='%.15g')
    return t

def build(esc):
    t=prepare();contracts=json.loads((ROOT/'source/table_contract.json').read_text())
    def emit(old,headers,rows,align,note=''):
        contract=contracts[old];name=RENAMES[old]
        half=old!='table_01_main_six_summary'
        lines=[r'\begin{minipage}[t]{0.49\linewidth}' if half else r'\begin{table}[H]',r'\centering\footnotesize',r'\captionsetup{type=table}',r'\captionof{table}{'+contract['caption']+'}',r'\label{'+contract['label']+'}',r'\setlength{\tabcolsep}{2.2pt}',r'\renewcommand{\arraystretch}{1.10}',r'\begin{tabularx}{\linewidth}{'+align+'}',r'\toprule', ' & '.join(headers)+r' \\',r'\midrule']
        lines+=[' & '.join(row)+r' \\' for row in rows]
        lines += [r'\bottomrule',r'\end{tabularx}']
        if note:lines += [r'\par\vspace{2pt}{\scriptsize '+note+'}']
        lines += [r'\end{minipage}' if half else r'\end{table}','']
        (OUT/(name+'.tex')).write_text('\n'.join(lines))
    emit('table_01_main_six_summary',['Dataset',r'\makecell{PCG risk\\(\%)}',r'\makecell{Best comparator\\risk (\%)}',r'\makecell{$\Delta_{\rm primary}$\\(pp)}',r'\makecell{Coverage\\(\%)}'],[[esc(row.dataset),f'{row.pcg_accepted_risk:.2f}',f'{row.best_applicable_risk:.2f}',f'{row.delta_primary_pp:+.2f}',f'{row.retained_coverage:.2f}'] for row in t.itertuples()],r'@{}Xrrrr@{}',r'Means over seven models per dataset; best applicable comparator selected within each cell. $\Delta_{\rm primary}=R_{\rm PCG}-R_{\rm best}$; negative favors PCG-MAS. No matched-coverage comparator estimate is inferred.')
    scope=pd.read_csv(DATA/'capability_scope_matrix.csv',keep_default_na=False)
    partial=set(map(tuple,scope.loc[scope.state.eq('PARTIAL'),['method','channel']].to_numpy()))
    assert partial=={('ShieldAgent','Recomputable'),('AgentRR','Recomputable')}
    cols=['Integrity','Replay','Policy','Support','Recomputable'];grid=scope.pivot(index='method',columns='channel',values='state').reindex(scope.method.drop_duplicates())[cols]
    emit('table_08_baseline_scope_matrix',['Method']+[r'\rotatebox{60}{'+c+'}' for c in cols],[[esc(m)]+[r'\scriptsize '+v for v in row] for m,row in grid.iterrows()],r'@{}Xccccc@{}')
    cost=pd.read_csv(DATA/'table_28_protocol_cost.csv');phases=pd.read_csv(DATA/'figure_D2_cost_telemetry.csv')
    np.testing.assert_allclose(cost.iloc[:,1],phases.share.dropna())
    emit('table_28_protocol_cost',['Phase','Share','Path'],[[esc(row.iloc[0]),f'{100*float(row.iloc[1]):g}'+r'\%', 'Conditional' if 'only' in row.iloc[2] else esc(row.iloc[2])] for _,row in cost.iterrows()],r'@{}>{\raggedright\arraybackslash}Xr>{\raggedright\arraybackslash}p{.25\linewidth}@{}',r'Normalized PCG-MAS resource shares; forensic replay applies to audited/failed cases, not every direct path.')
    audit=pd.read_csv(DATA/'table_03_audit_calibration_summary.csv')
    note='; '.join(f'{esc(row.Dataset)}: {row.N}' for row in audit.itertuples())
    if audit.N.nunique()==1:note=f'N = {audit.N.iloc[0]:,} per dataset.'
    emit('table_03_audit_calibration_summary',['Dataset','Prec.','Recall','FPR','ECE'],[[esc(row.Dataset)]+[f'{getattr(row,c):.3f}' for c in ['Precision','Recall','FPR','ECE']] for row in audit.itertuples()],r'@{}Xrrrr@{}',note)
    for old in RENAMES:
        p=OUT/(old+'.tex')
        if p.exists():p.unlink()
