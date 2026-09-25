"""Reduction advantage is PCG reduction minus comparator reduction.

For a common reference risk, this equals comparator risk minus PCG risk.
The primary risk endpoint remains PCG risk minus best-comparator risk.
"""
import numpy as np
import matplotlib.pyplot as plt
from figure_labels import direct_labels, MODEL_SHORT
from matplotlib.colors import TwoSlopeNorm
from render_result_figures import PALETTE, INK, GRAY, TEAL, title, tidy, heat

def _render(read, finish):
    s=read('method_summary.csv'); p=read('paired_cell_effects.csv'); raw=read('cell_metrics.csv')
    f=plt.figure(figsize=(16,13));f._final_layout=True
    plt.rcParams.update({"font.size":18,"axes.labelsize":18,"axes.titlesize":20,"xtick.labelsize":17,"ytick.labelsize":17})
    gs=f.add_gridspec(3,2,height_ratios=[1,1.15,.7],hspace=.60,wspace=.55,left=.15,right=.96,top=.94,bottom=.07)
    a=f.add_subplot(gs[0,0]); y=np.arange(len(s))
    # Legacy input field already stores comparator minus PCG. Do not negate it.
    reduction=s.risk_gap_vs_pcg_pp.to_numpy()
    lo=s.risk_gap_ci95_low_pp.to_numpy();hi=s.risk_gap_ci95_high_pp.to_numpy()
    a.barh(y,reduction,color=PALETTE,height=.57)
    a.errorbar(reduction,y,xerr=[reduction-lo,hi-reduction],fmt='none',ecolor=INK,capsize=4,lw=1.5)
    for v,upper,yy in zip(reduction,hi,y):
        a.annotate(f'{v:.2f}',(max(v,upper),yy),xytext=(7,0),textcoords='offset points',va='center',color='black')
    a.set_yticks(y,[m.replace('SignalMatchedFusion','SignalMatched\nFusion') for m in s.method]);a.invert_yaxis()
    a.set_xlim(min(0,float(lo.min())*1.2),max(float(hi.max())*1.26,.1))
    a.spines['left'].set_color(INK);a.axvline(0,color=INK,lw=1)
    a.set_xlabel('Accepted-harm reduction (pp)')
    title(a,'A','Headline effect');tidy(a)
    b=f.add_subplot(gs[0,1])
    tab=-p.pivot(index='model',columns='dataset',values='delta_pcg_minus_best_pp')
    im=heat(b,tab,'YlGnBu',fmt='.1f')
    b.set_xticks(np.arange(len(tab.columns)+1)-.5,minor=True)
    b.set_yticks(np.arange(len(tab.index)+1)-.5,minor=True)
    b.grid(which='minor',color='white',linewidth=1.6)
    b.tick_params(which='minor',bottom=False,left=False)
    im.set_clim(min(0,float(tab.min().min())),float(tab.max().max()))
    # Add the color scale after the six data axes to preserve their stable order.
    title(b,'B','Across backends')
    b.set_yticklabels([MODEL_SHORT[m] for m in tab.index])
    for txt in b.texts:txt.set_fontsize(16)
    if tab.min().min()<0<tab.max().max():im.set_cmap('RdBu');im.set_norm(TwoSlopeNorm(vcenter=0,vmin=tab.min().min(),vmax=tab.max().max()))
    c=f.add_subplot(gs[1,0]); points=[]
    for i,method in enumerate(s.method):
        cells=raw[(raw.method==method)&raw.applicable.astype(str).str.lower().eq('true')]
        c.scatter(cells.coverage,cells.risk,s=24,alpha=.14,color=PALETTE[i],edgecolors='none',zorder=2)
        row=s[s.method==method].iloc[0]
        c.scatter(row.mean_coverage,row.mean_risk,s=170 if method=='PCG-MAS' else 110,color=PALETTE[i],edgecolors=INK,linewidths=1.3,zorder=6)
        c.scatter(row.mean_coverage,row.mean_risk,s=60 if method=='PCG-MAS' else 30,color='white',zorder=7)
        points.append((row.mean_coverage,row.mean_risk,method,PALETTE[i]))
    c.set_xlim(raw.coverage.min()-3,raw.coverage.max()+3);c.set_ylim(max(0,raw.risk.min()-3),raw.risk.max()+5)
    title(c,'C','Risk vs coverage');c.set_xlabel('Retained coverage (%)');c.set_ylabel('Accepted risk (%)');tidy(c)
    direct_labels(c,points,fontsize=17)
    d=f.add_subplot(gs[1,1]);datasets=list(p.dataset.drop_duplicates());values=-p.delta_pcg_minus_best_pp
    if len(p)!=56 or not p.groupby('dataset').size().eq(7).all():raise ValueError('Expected seven models per dataset')
    d.axvline(0,color=INK,lw=1,zorder=1);adverse=[]
    for i,ds in enumerate(datasets):
        cells=p[p.dataset==ds];v=-cells.delta_pcg_minus_best_pp.to_numpy();q1,median,q3=np.quantile(v,[.25,.5,.75],method='linear')
        d.plot([v.min(),v.max()],[i,i],color=GRAY,lw=1.2,zorder=2)
        d.plot([q1,q3],[i,i],color=INK,lw=5,solid_capstyle='butt',zorder=3)
        d.scatter(v,i+np.linspace(-.10,.10,len(v)),color=GRAY,alpha=.55,s=30,edgecolors='white',linewidths=.4,zorder=4)
        d.scatter(median,i,marker='D',s=80,color=TEAL,edgecolors='white',linewidths=1,zorder=5)
        worst=np.argmin(v);name=MODEL_SHORT[cells.iloc[worst].model]
        adverse.append((v[worst],i,name,INK))
    d.set_yticks(range(8),datasets);d.set_ylim(7.5,-.75);d.set_xlim(min(0,values.min()-.2),values.max()+.6)
    d.set_xlabel('Accepted-harm reduction (pp)');title(d,'D','Across-model stability');tidy(d)
    direct_labels(d,adverse,fontsize=17)
    d.text(.99,.98,'◆ Median · thick IQR',transform=d.transAxes,ha='right',va='top',fontsize=12,color=INK)
    sub=gs[2,:].subgridspec(2,1,height_ratios=[1.5,1],hspace=.3)
    e=f.add_subplot(sub[0]);g=f.add_subplot(sub[1],sharex=e);x=np.arange(len(s))
    bars=e.bar(x,s.mean_token_multiplier,color=PALETTE,width=.53)
    for bar,value in zip(bars,s.mean_token_multiplier):
        e.annotate(f'{value:.2f}×',(bar.get_x()+bar.get_width()/2,value),xytext=(0,7),textcoords='offset points',ha='center',va='bottom',color='black')
    e.set_ylim(0,max(2.05,float(s.mean_token_multiplier.max())*1.25));e.set_yticks([0,1,2]);e.axhline(1,color=GRAY,ls='--',lw=1)
    e.set_ylabel('Tokens (×)',rotation=0,ha='right',va='center');e.tick_params(labelbottom=False)
    title(e,'E','Resource trade-off');tidy(e,'y')
    g.scatter(x,s.risk_reduction_vs_nocert_pp,c=PALETTE,marker='D',s=120,edgecolors='white',linewidths=1.3,zorder=3)
    g.scatter(x,s.risk_reduction_vs_nocert_pp,facecolors='none',edgecolors=INK,marker='o',s=235,linewidths=1.3,zorder=4)
    g.axhline(0,color=GRAY,lw=.8);g.set_ylim(-1.3,max(float(s.risk_reduction_vs_nocert_pp.max())*1.18,1))
    g.set_ylabel('Reduction\n(pp)',rotation=0,ha='right',va='center');g.set_xticks(x,[m.replace('SignalMatchedFusion','SignalMatched\nFusion') for m in s.method]);tidy(g,'y')
    cb=f.colorbar(im,ax=b,fraction=.047,pad=.04);cb.set_label('Reduction (pp)')
    box=b.get_position(); scale=1.0
    b.set_position([box.x1-box.width*scale,box.y1-box.height*scale+box.height*.1,box.width*scale,box.height*scale])
    cbbox=cb.ax.get_position()
    cb.ax.set_position([cbbox.x0,box.y1-box.height*scale+box.height*.1,cbbox.width,box.height*scale])
    for ax in f.axes:
        ax.tick_params(axis='both',labelsize=16)
    finish(f,1)

def render(read,finish):
    with plt.rc_context():
        _render(read,finish)
