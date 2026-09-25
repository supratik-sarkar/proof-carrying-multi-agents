"""Declared theory geometry and data-backed mechanism diagnostics."""
import json
from statistics import NormalDist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap,BoundaryNorm
from render_result_figures import INK,BLUE,TEAL,RED,GOLD,GRAY,METHOD_COLORS,title,tidy

DATASET_ORDER=['FEVER','HotpotQA','PubMedQA','TAT-QA','2Wiki','Mind2Web','BFCL V1','AgentDojo']
DATASET_COLORS=dict(zip(DATASET_ORDER,['#2374AB','#D77920','#258641','#AF4678','#775BA6','#846348','#BC5266','#148E9A']))

def config(renderer,name):
    renderer.READS.setdefault(renderer.ACTIVE,set()).add(name)
    return json.loads((renderer.DATA/name).read_text())

def selection_diagnostic(raw):
    applicable=raw[raw.applicable.astype(str).str.lower().eq('true')]
    keys=['model','dataset'];pcg=applicable[applicable.method=='PCG-MAS'];base=applicable[applicable.method=='NoCert']
    paired=pcg.merge(base,on=keys,suffixes=('_pcg','_base'),validate='one_to_one')
    assert len(paired)==56 and paired.groupby('dataset').size().eq(7).all()
    paired['coverage_change_pp']=paired.coverage_pcg-paired.coverage_base
    paired['risk_change_pp']=paired.risk_pcg-paired.risk_base
    means=paired.groupby('dataset')[['coverage_change_pp','risk_change_pp']].mean().reindex(DATASET_ORDER)
    return paired,means

def wilson_interval(p,n,confidence):
    """Two-sided Wilson score interval for a proportion at each sample size."""
    z=NormalDist().inv_cdf(.5+confidence/2)
    denominator=1+z*z/n
    center=(p+z*z/(2*n))/denominator
    half_width=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return center-half_width,center+half_width

def figure3(r):
    contract=config(r,'figure_03_capability_contract.json');cfg=config(r,'plot_parameters.json')
    rows=contract['methods'];cols=contract['channels'];entries={(x['method'],x['channel']):x['state'] for x in contract['entries']}
    states=np.array([[entries[m,c] for c in cols] for m in rows]);mapping={'N/A':0,'PARTIAL':1,'FULL':2}
    values=np.array([[mapping[s] for s in row] for row in states])
    f=plt.figure(figsize=(20,10.5));gs=f.add_gridspec(1,3,width_ratios=[1.55,1,1],left=.095,right=.98,bottom=.32,top=.71,wspace=.42)
    a,b,c=[f.add_subplot(gs[i]) for i in range(3)]
    colors=['#EDF0F3','#84BCD7',RED];a.imshow(values,cmap=ListedColormap(colors),norm=BoundaryNorm([-.5,.5,1.5,2.5],3),aspect='auto')
    a.set_yticks(range(len(rows)),rows);a.set_xticks(range(5),['Integrity','Replay','Policy','Support','Recomputable'],rotation=35,ha='right')
    for i in range(len(rows)):
        for j in range(5):a.text(j,i,states[i,j],ha='center',va='center',color='white' if values[i,j]==2 else INK,fontsize=13)
    a.set_xticks(np.arange(6)-.5,minor=True);a.set_yticks(np.arange(8)-.5,minor=True);a.grid(which='minor',color='white',lw=2);a.tick_params(which='minor',bottom=False,left=False)
    a.legend(handles=[Patch(color=colors[i],label=s) for i,s in [(2,'FULL'),(1,'PARTIAL'),(0,'N/A')]],loc='lower right',bbox_to_anchor=(1.04,1.015),ncol=3,frameon=False,columnspacing=.7,handlelength=1)
    title(a,'A','Declared capability scope');a.set_title('A  Declared capability scope',loc='left',y=1.28,fontweight='bold')
    dep=cfg['dependence'];k=np.linspace(dep['k_min'],dep['k_max'],240);rho=np.linspace(dep['rho_min'],dep['rho_max'],240);K,R=np.meshgrid(k,rho);Z=R**(K-1)*dep['epsilon']**K
    b.contourf(K,R,Z,levels=16,cmap='viridis_r')
    cs=b.contour(K,R,Z,levels=dep['contours'],colors='white',linewidths=2)
    # Place contour labels in the open region, away from the operating point.
    positions=[]
    for level,rr in zip(dep['contours'],[2.25,1.85,1.55]):
        kk=(np.log(level)+np.log(rr))/(np.log(rr)+np.log(dep['epsilon']));positions.append((kk,rr))
    labels=b.clabel(cs,fmt=lambda v:f'{v:.0%}',manual=positions,fontsize=17,inline=True,inline_spacing=8)
    for label in labels:label.set_bbox(dict(facecolor='#193948',edgecolor='white',boxstyle='round,pad=.2',alpha=.98));label.set_rotation(0)
    b.axhspan(dep['closed_rho'],dep['rho_max'],color=RED,alpha=.96,zorder=4)
    b.text(.5,.81,'GATE\nCLOSED',transform=b.transAxes,color='white',ha='center',va='center',fontweight='bold',zorder=5)
    b.axvline(dep['operating_k'],ls='--',color='white',lw=1.4);b.axhline(dep['operating_rho'],ls='--',color='white',lw=1.4)
    b.scatter(dep['operating_k'],dep['operating_rho'],s=220,facecolors='none',edgecolors='black',linewidths=2,zorder=6);b.scatter(dep['operating_k'],dep['operating_rho'],marker='+',s=180,color='white',lw=2,zorder=7)
    b.set_xlabel('Redundant branches k');b.set_ylabel('Dependence inflation factor ρ');b.set_title('B  Declared gate geometry',loc='left',y=1.28,fontweight='bold')
    au=cfg['audit_theory'];n=np.linspace(au['probes_min'],au['probes_max'],160)
    ci_low,ci_high=wilson_interval(au['residual'],n,au['confidence_level'])
    sampling=np.sqrt(np.log(au['inverse_delta'])/(2*n))
    for u,color in zip(au['uncovered_percent'],[BLUE,TEAL,GOLD,RED]):
        envelope=100*(au['residual']+sampling)+u
        c.plot(n,envelope,color=color,lw=2.8,label=f'{u:g}% uncovered')
        c.fill_between(n,100*(ci_low+sampling)+u,100*(ci_high+sampling)+u,color=color,alpha=.1)
    c.set_title('C  Audit envelope',loc='left',y=1.28,fontweight='bold');c.set_xlabel('Audit probes');c.set_ylabel('Upper envelope (%); lower is better');c.legend(loc='upper right',frameon=False);tidy(c)
    f.text(.52,.23,'Crosshair: declared\noperating point.\nρ is an inflation factor,\nnot correlation.',ha='left',va='top')
    f.text(.79,.23,'Uncovered mass: deployment\nstrata absent from the audit.\nMore probes: less uncertainty.\nMore missing mass: worse bound.\nShading: 95% Wilson\nconfidence intervals.',ha='left',va='top')
    f.text(.095,.06,'N/A: not established under this definition.\nRerunning a checker is not an\nexternally replayable acceptance contract.',ha='left',va='top')
    for ax in f.axes:ax.tick_params(labelsize=15)
    r.finish(f,3)

def render_decomposition(ax,sv,table_sv,protocol,r):
    if protocol.get('decomposition_supported') is not True or protocol.get('validated_non_degenerate') is not True:
        raise ValueError('Protocol does not authorize the supplied S/V decomposition')
    if protocol.get('component_data')!='selectivity_verification.csv' or protocol.get('tabular_crosscheck')!='table_25_protocol_sv.csv':
        raise ValueError('Protocol does not bind the authoritative S/V sources')
    if set(sv.dataset)!=set(DATASET_ORDER) or len(sv)!=8:raise ValueError('Incomplete S/V dataset coverage')
    vals=sv[['selection_pp','verification_pp','total_pp']].to_numpy(float)
    if not np.isfinite(vals).all() or np.allclose(vals[:,0],0) or np.allclose(vals[:,1],0):raise ValueError('Degenerate or missing S/V components')
    np.testing.assert_allclose(vals[:,0]+vals[:,1],vals[:,2],rtol=0,atol=1e-12)
    table=table_sv.set_index('Dataset').reindex(DATASET_ORDER)
    if table.isna().any().any():raise ValueError('Incomplete Table 25 S/V cross-check')
    np.testing.assert_allclose(sv.set_index('dataset').loc[DATASET_ORDER,'selection_pp'].round(2),table['S pp'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(sv.set_index('dataset').loc[DATASET_ORDER,'verification_pp'].round(2),table['V pp'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(sv.set_index('dataset').loc[DATASET_ORDER,'total_pp'].round(2),table['Delta pp'],rtol=0,atol=1e-12)
    shares=table['Verification share'].str.rstrip('%').astype(int).to_numpy()
    np.testing.assert_array_equal(np.rint(100*sv.set_index('dataset').loc[DATASET_ORDER,'verification_share']).astype(int),shares)
    sv=sv.set_index('dataset').reindex(DATASET_ORDER).reset_index()
    x=np.arange(len(sv));ax.bar(x,sv.selection_pp,color=BLUE,label='Selectivity');ax.bar(x,sv.verification_pp,bottom=sv.selection_pp,color=TEAL,label='Verification');ax.set_xticks(x,sv.dataset,rotation=35,ha='right');ax.legend();title(ax,'C','Validated S/V decomposition')

def figure4(r):
    ab=r.read('ablation_summary.csv');rc=r.read('risk_coverage_summary.csv');p=r.read('paired_cell_effects.csv');sv=r.read('selectivity_verification.csv');table_sv=r.read('table_25_protocol_sv.csv');protocol=config(r,'figure_04_protocol.json')
    f=plt.figure(figsize=(16,11.5));gs=f.add_gridspec(2,2,height_ratios=[1,1.15],left=.15,right=.97,top=.92,bottom=.14,hspace=.65,wspace=.58)
    a,b,c,d=[f.add_subplot(gs[i,j]) for i,j in [(0,0),(0,1),(1,0),(1,1)]]
    ab=ab[ab.dataset=='ALL_DATASETS'];y=np.arange(len(ab))*1.35;a.barh(y,ab.median_risk_increase_pp,color=TEAL,height=.7);a.set_yticks(y,ab.variant);a.invert_yaxis();a.set_xlabel('Median of dataset medians (pp)');title(a,'A','Component removal');tidy(a)
    for method,g in rc.groupby('method',sort=False):
        g=g.sort_values('threshold_index');b.plot(g.mean_coverage,g.mean_risk,label=method,color=METHOD_COLORS[method],lw=2);b.fill_between(g.mean_coverage,g.q15_risk,g.q85_risk,color=METHOD_COLORS[method],alpha=.09)
    summary=r.read('method_summary.csv')
    expected={'NoCert','MiniCheck','AlignScore','QAFactEval','CMVO','SignalMatchedFusion','PCG-MAS'}
    if set(summary.method)!=expected or len(summary)!=len(expected):raise ValueError('Incomplete or duplicate operating-point summary')
    for point in summary.itertuples():
        selected=point.method=='PCG-MAS';color=METHOD_COLORS[point.method]
        b.scatter(point.mean_coverage,point.mean_risk,s=150 if selected else 95,facecolors=color if selected else 'white',edgecolors=color,linewidths=1.8 if selected else 1.4,zorder=8,label='NoCert' if point.method=='NoCert' else '_nolegend_')
    b.set_xlabel('Retained coverage (%)');b.set_ylabel('Accepted risk (%)');title(b,'B','Risk and retained coverage');b.legend(ncol=2,frameon=False,loc='upper left',fontsize=12);tidy(b)
    render_decomposition(c,sv,table_sv,protocol,r)
    models=list(p.model.drop_duplicates())
    for i,model in enumerate(models):
        cells=p[p.model==model]
        for off,row in zip(np.linspace(-.23,.23,len(cells)),cells.itertuples()):d.scatter(-row.delta_pcg_minus_best_pp,i+off,s=100,color=DATASET_COLORS[row.dataset],edgecolors='white',linewidths=1.2,zorder=3)
    d.set_yticks(range(len(models)),models);d.set_ylim(len(models)-.5,-5.5)
    d.set_xlim(min(0,float((-p.delta_pcg_minus_best_pp).min())*1.1),float((-p.delta_pcg_minus_best_pp).max())*1.15)
    d.axvline(0,color=INK,lw=1);d.legend(handles=[Line2D([],[],ls='',marker='o',color=DATASET_COLORS[ds],label=ds,markersize=8) for ds in DATASET_ORDER],loc='upper right',framealpha=.95)
    title(d,'D','Reduction across model families')
    d.set_xlabel('Accepted-harm reduction advantage (pp)\nPositive favors PCG');tidy(d)
    for ax in f.axes:ax.tick_params(labelsize=13)
    r.finish(f,4)
