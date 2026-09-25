"""Recompute declared summaries from surviving data, without changing manuscript files."""
from pathlib import Path
import csv, hashlib, json, math, shutil, tempfile
from datetime import datetime,timezone
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'14_artifacts/manuscript_stack/data'
WORK=Path(tempfile.mkdtemp(prefix='pcg-metric-replay-'))
OUT=WORK/'recomputed';OUT.mkdir()
def load(n):return pd.read_csv(DATA/n,keep_default_na=False,na_values=['NA',''])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before={p.name:sha(p) for p in DATA.iterdir() if p.is_file()}
m=load('cell_metrics.csv');a=m[m.applicable]; keys=['model','dataset'];cols=['risk','coverage','utility','native_utility','token_multiplier']
pcg=a[a.method=='PCG-MAS'].set_index(keys);nocert=a[a.method=='NoCert'].set_index(keys)
built={};summ=[]
for method,g in a.groupby('method',sort=False):
 g=g.set_index(keys);gap=g.risk-pcg.risk.reindex(g.index);reduction=nocert.risk.reindex(g.index)-g.risk
 half=1.96*gap.std(ddof=1)/math.sqrt(len(gap))
 summ.append(dict(method=method,applicable_cells=len(g),**{'mean_'+c:g[c].mean() for c in cols},risk_gap_vs_pcg_pp=gap.mean(),risk_gap_ci95_low_pp=gap.mean()-half,risk_gap_ci95_high_pp=gap.mean()+half,risk_reduction_vs_nocert_pp=reduction.mean()))
built['method_summary.csv']=pd.DataFrame(summ)
paired=[]
for r in load('cell_registry_56.csv').itertuples():
 candidates=a[(a.model==r.model)&(a.dataset==r.dataset)&(a.method!='PCG-MAS')];best=candidates.sort_values(['risk','method']).iloc[0];p=pcg.loc[(r.model,r.dataset)]
 paired.append(dict(cell_id=r.cell_id,model=r.model,dataset=r.dataset,pcg_risk=p.risk,pcg_coverage=p.coverage,best_applicable_method=best.method,best_applicable_risk=best.risk,delta_pcg_minus_best_pp=p.risk-best.risk))
built['paired_cell_effects.csv']=pd.DataFrame(paired)
built['method_dataset_summary.csv']=pd.DataFrame([dict(method=method,dataset=ds,n_applicable_models=int(g.applicable.sum()),**{'mean_'+c:g.loc[g.applicable,c].mean() for c in cols}) for (method,ds),g in m.groupby(['method','dataset'],sort=False)])
r=load('risk_coverage_curves.csv')
built['risk_coverage_summary.csv']=pd.DataFrame([dict(method=method,threshold_index=t,n_datasets=g.dataset.nunique(),mean_coverage=g.coverage.mean(),mean_risk=g.risk.mean(),q15_risk=g.risk.quantile(.15),q85_risk=g.risk.quantile(.85)) for (method,t),g in r.groupby(['method','threshold_index'],sort=False)])
z=load('ablation_samples.csv');ab=[dict(variant=v,dataset=d,n=len(g),mean_risk_increase_pp=g.risk_increase_pp.mean(),median_risk_increase_pp=g.risk_increase_pp.median()) for (v,d),g in z.groupby(['variant','dataset'],sort=False)]
for v,g in pd.DataFrame(ab).groupby('variant',sort=False):ab.append(dict(variant=v,dataset='ALL_DATASETS',n=g.n.sum(),mean_risk_increase_pp=g.mean_risk_increase_pp.mean(),median_risk_increase_pp=g.median_risk_increase_pp.median()))
built['ablation_summary.csv']=pd.DataFrame(ab)
p=json.loads((DATA/'plot_parameters.json').read_text())
cap=pd.DataFrame([dict(method=method,channel=ch,level=level,state='FULL' if level==1 else ('PARTIAL' if level>0 else 'N/A')) for method,levels in zip(p['capability_methods'],p['capability_matrix']) for ch,level in zip(p['capability_channels'],levels)])
built['capability_scope_matrix.csv']=cap
w=load('witness_matrix.csv'); mapping={'W_H':'Integrity','W_Pi':'Replay','W_Gamma':'Policy','W_entail':'Support'}
w['mapped_channel']=w.witness.map(mapping)
assert w.mapped_channel.notna().all(),w.witness.unique()
w=w.merge(cap[['method','channel','state']],left_on=['method','mapped_channel'],right_on=['method','channel'],validate='many_to_one').drop(columns='channel').rename(columns={'state':'capability_state'})
w['native_scope']=w.capability_state!='N/A';w['plot_value']=w.accept_rate.where(w.native_scope)
built['witness_matrix_scoped.csv']=w
channel={'- $V_H$':'V_H','- $V_\\Gamma$':'V_Gamma','- $V_\\Pi$':'V_Pi','- $V_\\vdash$':'V_entail'}
f=[]
for _,r in load('table_07_channel_ablation.csv').iterrows():
 if r['Variant'] in channel:f.append(dict(panel='online_certificate_channels',channel=channel[r['Variant']],metric='risk_increase_pp',value=float(r['Delta risk pp']),dispersion=float(r['Dispersion']),source_file='table_07_channel_ablation.csv'))
for r in load('table_10_replay_drift_covgap.csv').itertuples():f.append(dict(panel='audit_sidecar',channel=r.Channel,metric='rate_pct',value=float(r.Rate.rstrip('%')),dispersion=np.nan,source_file='table_10_replay_drift_covgap.csv'))
built['figure_C1_certificate_failure_channel_atlas.csv']=pd.DataFrame(f)
built['timing_summary.csv']=pd.DataFrame([dict(method=method,n=len(g),median_latency_s=g.latency_s.median(),p95_latency_s=g.latency_s.quantile(.95)) for method,g in load('timing_samples.csv').groupby('method',sort=False)])
cost=[];means=a.groupby('method',sort=False).token_multiplier.mean()
telemetry=load('cost_telemetry.csv')
for method,t in means.items():
 assert np.allclose(telemetry.loc[telemetry.method==method,'token_multiplier'],t,rtol=1e-11,atol=1e-11)
phase_values=load('table_28_protocol_cost.csv')['Share of PCG direct cost'].astype(float).to_numpy()
assert np.isclose(phase_values.sum(),1.0)
assert np.allclose(telemetry.loc[telemetry.method=='PCG-MAS','share'].to_numpy(),phase_values)
for method,t in means.items():cost.append(dict(row_type='METHOD_TOTAL',method=method,phase=np.nan,share=np.nan,token_multiplier=t,absolute_normalized_cost=t,path=np.nan))
for _,r in load('table_28_protocol_cost.csv').iterrows():cost.append(dict(row_type='PCG_PHASE',method='PCG-MAS',phase=r['Phase'],share=float(r['Share of PCG direct cost']),token_multiplier=means['PCG-MAS'],absolute_normalized_cost=float(r['Share of PCG direct cost'])*means['PCG-MAS'],path=r['Path']))
built['figure_D2_cost_telemetry.csv']=pd.DataFrame(cost)
identities={'method_summary.csv':['method'],'paired_cell_effects.csv':['cell_id'],'method_dataset_summary.csv':['method','dataset'],'risk_coverage_summary.csv':['method','threshold_index'],'ablation_summary.csv':['variant','dataset'],'capability_scope_matrix.csv':['method','channel'],'witness_matrix_scoped.csv':['method','witness'],'figure_C1_certificate_failure_channel_atlas.csv':['panel','channel'],'timing_summary.csv':['method'],'figure_D2_cost_telemetry.csv':['row_type','method','phase']}
checks=[]; failures=[]
for name,df in built.items():
 expected=load(name);idcols=identities[name]
 left=df.sort_values(idcols,na_position='last').reset_index(drop=True);right=expected.sort_values(idcols,na_position='last').reset_index(drop=True)
 errors=[];maximum=0.
 if len(left)!=len(right):errors.append('row count differs')
 else:
  for c in right:
   x=left[c];y=right[c]
   if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
    xv=pd.to_numeric(x).to_numpy(dtype=float);yv=y.to_numpy(dtype=float)
    if not np.allclose(xv,yv,rtol=1e-11,atol=1e-11,equal_nan=True):errors.append(c)
    finite=np.isfinite(xv)&np.isfinite(yv)
    if finite.any():maximum=max(maximum,float(np.abs(xv[finite]-yv[finite]).max()))
   elif not x.fillna('<NA>').astype(str).equals(y.fillna('<NA>').astype(str)):errors.append(c)
 df[right.columns].to_csv(OUT/name,index=False,na_rep='NA')
 checks.append({'file':name,'rows':len(df),'status':'FAIL' if errors else 'PASS','different_columns':errors,'max_absolute_numeric_difference':maximum,'source_sha256':sha(DATA/name),'recomputed_sha256':sha(OUT/name)})
 failures.extend([name+': '+e for e in errors])
assert before=={p.name:sha(p) for p in DATA.iterdir() if p.is_file()}
report={'replay_recorded_at_utc':datetime.now(timezone.utc).isoformat(),'status':'FAIL' if failures else 'PASS','reference':'Embedded v3.7 manuscript data authority.','scope':'Ten deterministic aggregate derivations; current per-observation values are explicitly not retained.','relative_tolerance':1e-11,'absolute_tolerance':1e-11,'checks':checks,'data_files_changed':0}
print(json.dumps(report,indent=2))
shutil.rmtree(WORK)
raise SystemExit(bool(failures))
