"""Render declared figure inputs without estimating new scientific summaries.

Explicit display sign conventions and evaluation of declared theory formulas
are the only scientific transformations; stored primary endpoints remain fixed.
Jitter, axis padding, colors and text offsets affect presentation only.
"""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import json
import textwrap
from matplotlib.text import Text
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import ListedColormap
from figure_contract import ROOT,NAMES,INPUTS,STATIC

DATA=Path(os.environ.get('PCG_MANUSCRIPT_DATA',ROOT/'data'))
OUT=Path(os.environ.get('PCG_MANUSCRIPT_IMAGES',ROOT/'images'))
OUT.mkdir(parents=True,exist_ok=True)
INK='#183044';BLUE='#2374AB';TEAL='#008978';RED='#C53645';GOLD='#D39522';GRAY='#8495A4';PURPLE='#7356A6'
PALETTE=['#53616D','#7C9DAE','#9B89A5','#91AAA0','#B39775','#887E99',TEAL]
METHOD_COLORS=dict(zip(['NoCert','MiniCheck','AlignScore','QAFactEval','CMVO','SignalMatchedFusion','PCG-MAS'],PALETTE))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B8C4CB','text.color':INK,'axes.labelcolor':INK,'pdf.fonttype':42,'savefig.facecolor':'white'})
READS={};ACTIVE=None

def read(name):
    p=DATA/name
    READS.setdefault(ACTIVE,set()).add(name)
    return pd.read_csv(p,keep_default_na=True)

def title(ax,letter,head,sub=''):
    ax.set_title(letter+'  '+head+('\n'+sub if sub else ''),loc='left',fontweight='bold',pad=12,linespacing=1.5)

def canvas(nrows=1,ncols=1,size=(11,4),**kw):
    f,a=plt.subplots(nrows,ncols,figsize=size,layout='constrained',**kw)
    f.set_constrained_layout_pads(w_pad=.08,h_pad=.09,wspace=.12,hspace=.15)
    return f,a

def tidy(ax,axis='x'):
    ax.set_axisbelow(True);ax.grid(axis=axis,color='#E8EDF0',lw=.7)

def finish(f,n):
    # Size labels for the actual manuscript width, including the paired minipages.
    target_width=2.695 if n in {18,19} else 5.5
    minimum=6.1*f.get_figwidth()/target_width
    for text in f.findobj(match=Text):
        if text.get_text() and not getattr(f,'_final_layout',False): text.set_fontsize(max(text.get_fontsize(),minimum))
    f.canvas.draw()
    for ax in f.axes:
        head=ax._left_title
        if head.get_text():
            width=ax.get_window_extent().width / f.dpi * 72
            count=max(18,int(width/(head.get_fontsize()*.53)))
            head.set_text('\n'.join(textwrap.fill(line,count,break_long_words=False) for line in head.get_text().split('\n')))
    f.savefig(OUT/(NAMES[n-1]+'.pdf'),metadata={'CreationDate':None,'ModDate':None},bbox_inches='tight')
    f.savefig(OUT/(NAMES[n-1]+'.png'),dpi=160,bbox_inches='tight')
    plt.close(f)

def labels(ax,values,ys,fmt='{:.2f}',color=INK):
    for v,y in zip(values,ys):
        if pd.isna(v):ax.text(.02,y,'N/A',transform=ax.get_yaxis_transform(),va='center',fontsize=8)
        else: ax.annotate(fmt.format(0 if v==0 else v),(v,y),xytext=(5 if v>=0 else -5,0),textcoords='offset points',ha='left' if v>=0 else 'right',va='center',fontsize=8,color=color)
    ax.margins(x=.25)

def heat(ax,frame,cmap='YlGnBu',fmt='.1f',zero=False,states=None):
    values=frame.to_numpy(dtype=float)
    finite=values[np.isfinite(values)]
    opt={}
    if zero:
        limit=max(abs(finite).max() if finite.size else 1,.01);opt={'vmin':-limit,'vmax':limit}
    cm=plt.get_cmap(cmap).copy();cm.set_bad('#EDF0F3')
    im=ax.imshow(np.ma.masked_invalid(values),cmap=cm,aspect='auto',**opt)
    ax.set_xticks(range(len(frame.columns)),[str(v).replace('SignalMatchedFusion','SignalMatched\nFusion') for v in frame.columns],rotation=35,ha='right')
    ax.set_yticks(range(len(frame.index)),frame.index)
    for i in range(len(frame.index)):
        for j in range(len(frame.columns)):
            v=values[i,j]
            if pd.isna(v):
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,hatch='///',edgecolor='#C8D0D7',lw=0));text='N/A';c=INK
            else:
                rgb=im.cmap(im.norm(v))[:3];c='white' if np.dot(rgb,[.2126,.7152,.0722])<.5 else INK;text=format(v,fmt)
            if states is not None:text=states.iloc[i,j]
            ax.text(j,i,text,ha='center',va='center',fontsize=7,color=c)
    return im

def paired_points(ax,d,group,letter,head):
    groups=list(d[group].drop_duplicates())
    for i,g in enumerate(groups):
        a=d[d[group]==g];offset=np.linspace(-.18,.18,len(a))
        ax.scatter(a.delta_pcg_minus_best_pp,i+offset,s=20,color=plt.get_cmap('tab10')(i%10),alpha=.85,edgecolors='white',linewidths=.3)
    ax.set_yticks(range(len(groups)),groups);ax.invert_yaxis();ax.axvline(0,color=GRAY,lw=.8)
    ax.set_xlabel('PCG − comparator (pp); lower is safer');title(ax,letter,head,'One point per declared cell');tidy(ax)

def fig1():
    from figure_01_layout import render
    render(read, finish)


def fig3():
    import sys
    from figure_03_04_layouts import figure3
    figure3(sys.modules[__name__])

def fig4():
    import sys
    from figure_03_04_layouts import figure4
    figure4(sys.modules[__name__])


def fig5():
    import sys
    from appendix_figure_layouts import figure5
    figure5(sys.modules[__name__])


def fig11():
    import sys
    from appendix_figure_layouts import figure11
    figure11(sys.modules[__name__])


def fig12():
    d=read('ablation_summary.csv');a=d.pivot(index='variant',columns='dataset',values='median_risk_increase_pp');order=list(d.variant.drop_duplicates());a=a.reindex(order);cols=[c for c in a.columns if c!='ALL_DATASETS']+['ALL_DATASETS'];a=a[cols];a=a.rename(columns={'ALL_DATASETS':'Dataset-median\nsummary'})
    f,ax=canvas(size=(11,4));im=heat(ax,a,'RdBu_r',zero=True);f.colorbar(im,ax=ax,fraction=.04,pad=.15).set_label('Risk increase (pp)');title(ax,'A','Ablation risk increase (pp)','Certificate-channel removals and system perturbations; larger positive = greater loss of safety')
    # Group boundaries follow labels in the supplied rows.
    channel=[i for i,v in enumerate(order) if v.startswith('- $V') or v.startswith('−V')]
    if channel:
        ax.axhline(min(channel)-.5,color=INK,lw=1.2);ax.axhline(max(channel)+.5,color=INK,lw=2)
        ax.text(1.015,.7,'Channel\nremovals',transform=ax.transAxes,fontsize=8,va='center');ax.text(1.015,.22,'System /\nalgorithm',transform=ax.transAxes,fontsize=8,va='center')
    ax.axvline(len(a.columns)-1.5,color=INK,lw=2);finish(f,12)

def fig13():
    d=read('audit_sampling.csv');f,axs=canvas(1,4,(13,3.7),sharey=True);y=np.arange(len(d))
    for ax,col,letter,head,unit in zip(axs[:3],['worst_stratum_coverage','envelope','ess'],'ABC',['Coverage ↑','Envelope ↓','Effective sample size'],['Worst-stratum (%)','Certified bound (%)','Weighted sample count']):
        ax.hlines(y,0,d[col],color='#CCDCE5',lw=5);ax.scatter(d[col],y,s=50,color=TEAL);labels(ax,d[col],y,'{:.1f}');ax.set_xlabel(unit);title(ax,letter,head);tidy(ax)
    axs[0].set_yticks(y,d.design.str.replace(' ','\n',n=1));axs[0].invert_yaxis();ax=axs[3];ax.barh(y-.15,d.violation,height=.25,color=BLUE,label='Violation');ax.barh(y+.15,d.false_cert,height=.25,color=RED,label='False certification');title(ax,'D','Failure rates ↓');ax.set_xlabel('Rate (%)');ax.legend(fontsize=7,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.2));tidy(ax);finish(f,13)

def fig14():
    d=read('witness_matrix_scoped.csv');a=d.pivot(index='method',columns='witness',values='plot_value').reindex(d.method.drop_duplicates());f,ax=canvas(size=(10,4));im=heat(ax,a,'RdYlGn_r');f.colorbar(im,ax=ax,fraction=.04,pad=.04).set_label('Acceptance rate (%)');ax.set_xticklabels([{'W_H':r'$W_H$','W_Pi':r'$W_\Pi$','W_Gamma':r'$W_\Gamma$','W_entail':r'$W_\vdash$'}[v] for v in a.columns],rotation=0);title(ax,'A','Separating witnesses: acceptance (%)','Hatched N/A = outside native scope; partial native scope is retained')
    if 'PCG-MAS' in a.index:ax.add_patch(Rectangle((-.5,list(a.index).index('PCG-MAS')-.5),len(a.columns),1,fill=False,edgecolor=INK,lw=2))
    ax.set_xlabel('Each witness targets its named certificate conjunct; lower PCG acceptance is desired');finish(f,14)

def fig15():
    d=read('injection_stress.csv');f,axs=canvas(1,2,(12,4),gridspec_kw={'width_ratios':[1.4,1]});x=np.arange(len(d));ax=axs[0]
    for offset,col,c,label in [(-.24,'attack_success',BLUE,'Attack success'),(0,'accepted_attack',RED,'Accepted attack'),(.24,'detection',TEAL,'Detection')]:ax.bar(x+offset,d[col],width=.23,color=c,label=label)
    ax.set_xticks(x,d.regime.str.replace(' ','\n',n=1));ax.set_ylabel('Rate (%)');title(ax,'A','Verifier isolation matters','Attack success, accepted attack, detection');ax.legend(fontsize=8,ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.2));tidy(ax,'y')
    ax=axs[1];y=np.arange(len(d));ax.hlines(y,0,d.rho_ucb,color='#D6E2E9',lw=7);ax.scatter(d.rho_ucb,y,c=[RED if 'Shared' in r else BLUE for r in d.regime],s=55);ax.set_yticks(y,d.regime.str.replace(' ','\n',n=1));ax.invert_yaxis();labels(ax,d.rho_ucb,y);ax.set_xlabel('Supplied dependence-factor upper bound');title(ax,'B','Common-mode dependence','Larger inflation weakens redundancy');tidy(ax);finish(f,15)

def fig16():
    d=read('shift_stress.csv');f,axs=canvas(1,2,(11,4))
    for i,(mode,g) in enumerate(d.groupby('mode',sort=False)):
        g=g.sort_values('severity');c=[GRAY,TEAL,BLUE][i];axs[0].plot(g.severity,g.bound_violation,color=c,label=mode,lw=2)
        axs[1].plot(g.utility,g.audit_coverage,color=c,lw=1,label=mode);sc=axs[1].scatter(g.utility,g.audit_coverage,c=g.alarm_power,cmap='viridis',vmin=d.alarm_power.min(),vmax=d.alarm_power.max(),s=25)
        start=g.iloc[0];end=g.iloc[-1];axs[1].annotate('',xy=(end.utility,end.audit_coverage),xytext=(g.iloc[-2].utility,g.iloc[-2].audit_coverage),arrowprops={'arrowstyle':'->','color':c,'lw':2});axs[1].annotate('high shift',(end.utility,end.audit_coverage),xytext=(4,-10-i*8),textcoords='offset points',fontsize=7,color=c)
    title(axs[0],'A','Validity under increasing shift','Scenario trajectories; lower violation is better');axs[0].set_xlabel('Shift severity');axs[0].set_ylabel('Bound violation (%)');axs[0].legend(fontsize=7,frameon=False);tidy(axs[0]);title(axs[1],'B','Utility and audit coverage','Arrowheads point toward higher shift');axs[1].set_xlabel('Utility (%)');axs[1].set_ylabel('Audit coverage (%)');f.colorbar(sc,ax=axs[1],label='Alarm power (%)',shrink=.75);tidy(axs[1]);finish(f,16)

def fig17():
    d=read('responsibility.csv');f,axs=canvas(1,3,(11,3),sharey=True,gridspec_kw={'width_ratios':[1.6,1,1]});y=np.arange(len(d));ax=axs[0];ax.hlines(y,d.top1,d.top3,color='#B9D8DB',lw=6);ax.scatter(d.top1,y,color=BLUE,s=50,label='Top-1');ax.scatter(d.top3,y,color=TEAL,s=50,label='Top-3');ax.set_yticks(y,d.regime.str.replace(' ','\n',n=1));ax.invert_yaxis();ax.set_xlabel('Diagnostic accuracy (%)');title(ax,'A','Allowing more diagnoses');ax.legend(fontsize=8,frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.2))
    for ax,col,letter,head,c in [(axs[1],'unresolved','B','Unresolved',GOLD),(axs[2],'median_margin','C','Responsibility margin',PURPLE)]:
        ax.barh(y,d[col],color=c,height=.45);labels(ax,d[col],y);ax.set_xlabel('Rate (%)' if col=='unresolved' else 'Supplied median margin');title(ax,letter,head);tidy(ax)
    tidy(axs[0]);finish(f,17)

def fig18():
    d=read('privacy_frontier.csv').sort_values('privacy_parameter');f,axs=canvas(2,1,(5.7,5.6));ax=axs[0];ax.errorbar(d.modelled_utility,d.modelled_risk,yerr=d.uncertainty,color=PURPLE,fmt='o-',capsize=3);ax.set_xlabel('Modelled utility (%)');ax.set_ylabel('Modelled risk (%)');title(ax,'A','Privacy–utility frontier','Whiskers: supplied uncertainty magnitude');tidy(ax)
    ax=axs[1];ax.plot(d.privacy_parameter,d.uncertainty,'o-',color=TEAL);ax.set_xscale('log',base=2);ax.set_xticks(d.privacy_parameter,[f'{x:g}' for x in d.privacy_parameter]);ax.set_xlabel('Declared privacy parameter');ax.set_ylabel('Uncertainty magnitude');title(ax,'B','Uncertainty across settings');tidy(ax);finish(f,18)

def fig19():
    d=read('scaling_surface.csv');a=d.pivot(index='support_size',columns='redundancy',values='cost_multiplier');f,ax=canvas(size=(5.7,4));heat(ax,a,'cividis',fmt='.2f');ax.set_xticklabels(a.columns,rotation=0);ax.set_xlabel('Redundant branches k');ax.set_ylabel('Support size');title(ax,'A','Modelled resource surface','Cost multiplier; lower means less resource use');finish(f,19)

def fig20():
    d=read('timing_samples.csv');s=read('timing_summary.csv');f,ax=canvas(size=(9,3.6));y=np.arange(len(s))
    for i,row in enumerate(s.itertuples()):
        a=d[d.method==row.method].latency_s.to_numpy();offset=((np.arange(len(a))*0.61803398875)%1-.5)*.34
        ax.scatter(a,i+offset,s=5,color=PALETTE[i],alpha=.25,rasterized=False);ax.plot([row.median_latency_s,row.p95_latency_s],[i,i],color=PALETTE[i],lw=5,alpha=.65);ax.scatter(row.median_latency_s,i,color=INK,s=23,zorder=4);ax.plot(row.p95_latency_s,i,marker='|',color=RED,ms=12,mew=2)
    ax.set_yticks(y,s.method);ax.invert_yaxis();ax.set_xlabel('Wall-clock latency (s); lower is faster');title(ax,'A','Run-level latency distribution','Raw samples; black median; red p95; no density smoothing');tidy(ax);finish(f,20)

def fig21():
    import sys
    from appendix_figure_layouts import figure21
    figure21(sys.modules[__name__])

def fig22():
    import sys
    from main_result_layouts import release_funnel
    release_funnel(sys.modules[__name__])

def fig23():
    import sys
    from main_result_layouts import robustness
    robustness(sys.modules[__name__])

FUNCTIONS={22:fig22,23:fig23,1:fig1,3:fig3,4:fig4,5:fig5,11:fig11,12:fig12,13:fig13,14:fig14,15:fig15,16:fig16,17:fig17,18:fig18,19:fig19,20:fig20,21:fig21}
def generate(n):
    global ACTIVE
    ACTIVE=n;FUNCTIONS[n]()
    if READS[n]!=set(INPUTS[n]):raise ValueError(f'Input declaration mismatch for figure {n}')

if __name__=='__main__':
    import sys
    for n in ([int(v) for v in sys.argv[1:]] or list(FUNCTIONS)):generate(n)
