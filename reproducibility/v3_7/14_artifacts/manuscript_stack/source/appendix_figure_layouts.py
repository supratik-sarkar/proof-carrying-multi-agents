"""Appendix layouts with explicit units, source checks and stable display semantics."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from render_result_figures import INK,BLUE,TEAL,RED,GRAY,PURPLE,title,tidy,heat,labels

def figure5(r):
    p=r.read('paired_cell_effects.csv');d=r.read('method_dataset_summary.csv')
    f,axes=plt.subplots(2,2,figsize=(16,14.4));f.subplots_adjust(left=.17,right=.96,top=.92,bottom=.13,hspace=1.1,wspace=.85)
    tabs=[p.pivot(index='model',columns='dataset',values='pcg_risk'),d.pivot(index='method',columns='dataset',values='mean_risk'),d.pivot(index='method',columns='dataset',values='mean_coverage')]
    ims=[]
    for ax,tab,letter,heading in zip(axes.flat,tabs,'ABC',['PCG accepted risk (%)','Method accepted risk (%)','Retained coverage (%)']):
        im=heat(ax,tab,'YlGnBu');ims.append((ax,im,heading))
        ax.set_xticks(np.arange(tab.shape[1]+1)-.5,minor=True);ax.set_yticks(np.arange(tab.shape[0]+1)-.5,minor=True);ax.grid(which='minor',color='white',lw=2);ax.tick_params(which='minor',bottom=False,left=False);title(ax,letter,heading)
    ax=axes[1,1]
    for i,model in enumerate(p.model.drop_duplicates()):
        cells=p[p.model==model];y=i+np.linspace(-.21,.21,len(cells));color=plt.get_cmap('tab10')(i)
        ax.scatter(-cells.delta_pcg_minus_best_pp,y,s=85,color=color,edgecolors='white',linewidths=1.1)
    ax.set_yticks(range(p.model.nunique()),p.model.drop_duplicates());ax.invert_yaxis()
    for i,t in enumerate(ax.get_yticklabels()):t.set_color(plt.get_cmap('tab10')(i))
    ax.axvline(0,color=INK,lw=1);ax.set_xlim(min(0,float((-p.delta_pcg_minus_best_pp).min())*1.1),float((-p.delta_pcg_minus_best_pp).max())*1.15)
    title(ax,'D','Cell reduction advantage','PCG minus best comparator\nIn accepted-harm reduction (pp)');ax.set_xlabel('Accepted-harm reduction advantage (pp)\nPositive favors PCG; color identifies model');tidy(ax)
    for ax,im,label in ims:
        cb=f.colorbar(im,ax=ax,fraction=.05,pad=.035);cb.set_label(label)
        box=ax.get_position();scale=1.25
        ax.set_position([box.x1-box.width*scale,box.y1-box.height*scale,box.width*scale,box.height*scale])
        cbox=cb.ax.get_position();cb.ax.set_position([cbox.x0,box.y1-box.height*scale,cbox.width,box.height*scale])
    for ax in f.axes:ax.tick_params(labelsize=12)
    r.finish(f,5)

CHANNELS=['V_H','V_Pi','V_Gamma','V_entail']
MATH=[r'$V_H$',r'$V_\Pi$',r'$V_\Gamma$',r'$V_\vdash$']
def checked_channels(r):
    d=r.read('figure_C1_certificate_failure_channel_atlas.csv');a=d[d.panel=='online_certificate_channels'].set_index('channel').loc[CHANNELS];b=d[d.panel=='audit_sidecar'].set_index('channel').loc[['ReplayFail','DriftFail','CovGap']]
    assert len(d)==7 and len(a)==4 and len(b)==3
    assert a.metric.eq('risk_increase_pp').all() and b.metric.eq('rate_pct').all()
    upstream=r.read('table_07_channel_ablation.csv').set_index('Variant');audit=r.read('table_10_replay_drift_covgap.csv').set_index('Channel')
    for key,symbol in zip(CHANNELS,MATH):
        row=upstream.loc['- '+symbol];np.testing.assert_allclose([a.loc[key,'value'],a.loc[key,'dispersion']],[row['Delta risk pp'],row['Dispersion']]);assert a.loc[key,'source_file']=='table_07_channel_ablation.csv'
    for key in b.index:assert np.isclose(b.loc[key,'value'],float(audit.loc[key,'Rate'].rstrip('%')));assert b.loc[key,'source_file']=='table_10_replay_drift_covgap.csv'
    return a,b

def figure11(r):
    a,b=checked_channels(r);f,axs=plt.subplots(1,2,figsize=(13,5));f.subplots_adjust(left=.1,right=.97,top=.73,bottom=.18,wspace=.75)
    for ax,data,letter,head,cats,xlab in [(axs[0],a,'A','Online acceptance-channel ablation',MATH,'Accepted-risk increase (pp)'),(axs[1],b,'B','Separate audit-sidecar failure channels',[r'$\mathsf{ReplayFail}$',r'$\mathsf{DriftFail}$',r'$\mathsf{CovGap}$'],'Aggregate audit rate (%)')]:
        y=np.arange(len(data));ax.barh(y,data.value,color=TEAL if letter=='A' else BLUE,height=.55);ax.set_yticks(y,cats);ax.invert_yaxis();labels(ax,data.value,y,'{:.1f}');ax.set_xlabel(xlab);title(ax,letter,head,'Aggregate value; no dataset split');tidy(ax)
    for ax in f.axes:ax.tick_params(labelsize=13)
    r.finish(f,11)

def checked_cost(r):
    d=r.read('figure_D2_cost_telemetry.csv');p=d[d.row_type=='PCG_PHASE'].copy();total=float(d[(d.row_type=='METHOD_TOTAL')&(d.method=='PCG-MAS')].absolute_normalized_cost.item())
    upstream=r.read('table_28_protocol_cost.csv');raw=r.read('cell_metrics.csv');tele=r.read('cost_telemetry.csv')
    phases=['Generation','Retrieval/canonicalization','Certificate assembly','Semantic verifier','Forensic replay']
    assert p.phase.tolist()==phases and len(p)==5 and p.method.eq('PCG-MAS').all()
    assert p.path.tolist()==['Always on']*4+['Audited/failed cases only']
    assert upstream.Phase.tolist()==phases and upstream.Path.tolist()==p.path.tolist()
    np.testing.assert_allclose(p.share,upstream['Share of PCG direct cost']);np.testing.assert_allclose(p.share.sum(),1)
    np.testing.assert_allclose(p.token_multiplier,total);np.testing.assert_allclose(p.absolute_normalized_cost,p.share*total)
    np.testing.assert_allclose(p.absolute_normalized_cost.sum(),total)
    cell=raw[(raw.method=='PCG-MAS')&raw.applicable.astype(str).str.lower().eq('true')];np.testing.assert_allclose(cell.token_multiplier,total)
    telemetry=tele[tele.method=='PCG-MAS'].set_index('phase').loc[['Generation','Retrieval','Certificate','Semantic verifier','Forensic replay']]
    np.testing.assert_allclose(telemetry.share,p.share);np.testing.assert_allclose(telemetry.token_multiplier,total)
    return p,total

def figure21(r):
    p,total=checked_cost(r);colors=[INK,BLUE,TEAL,'#CC79A7',GRAY]
    f,(a,b)=plt.subplots(2,1,figsize=(13,8.5));f.subplots_adjust(left=.18,right=.97,top=.83,bottom=.15,hspace=1.25)
    left=0
    for row,color in zip(p.itertuples(),colors):
        conditional=row.path!='Always on'
        a.barh(0,row.absolute_normalized_cost,left=left,height=.48,color=color,edgecolor='white',linewidth=1.5,hatch='////' if conditional else None)
        left+=row.absolute_normalized_cost
    a.axvline(1,color=GRAY,lw=1.5,ls='--');a.annotate('NoCert token reference',(1,.25),xytext=(0,18),textcoords='offset points',ha='center');a.text(total+.025,0,f'{total:.2f}×',va='center',fontweight='bold')
    a.set_xlim(0,total*1.18);a.set_ylim(-.55,.6);a.set_yticks([0],['PCG-MAS']);a.set_xlabel('Normalized PCG-MAS resource contributions (NoCert units)');title(a,'A','Anatomy of PCG-MAS direct resource cost');tidy(a)
    a.legend(handles=[Patch(facecolor=co,label=phase.replace('Retrieval/canonicalization','Retrieval / canonicalization'),hatch='////' if i==4 else None) for i,(co,phase) in enumerate(zip(colors,p.phase))],loc='upper center',bbox_to_anchor=(.48,-.38),ncol=3,frameon=False)
    left=0
    for row,color in zip(p.itertuples(),colors):
        conditional=row.path!='Always on';y=1 if conditional else 0
        b.barh(y,row.share*100,left=0 if conditional else left,height=.44,color=color,edgecolor='white',linewidth=1.5,hatch='////' if conditional else None)
        if not conditional:
            b.text(left+row.share*50,y,f'{row.share:.0%}',ha='center',va='center',color='white' if color!= '#CC79A7' else INK)
            left+=row.share*100
    online=float(p.loc[p.path=='Always on','share'].sum());forensic=float(p.loc[p.path!='Always on','share'].sum())
    b.text(online*100+1,0,f'{online:.0%}',va='center');b.text(forensic*100+1,1,f'{forensic:.0%} — audited / failed cases only',va='center')
    b.set_xlim(0,112);b.set_ylim(1.5,-.5);b.set_yticks([0,1],['Always-on\ncertification','Conditional\nforensic replay']);b.set_xlabel('Share of PCG-MAS resource envelope (%)');title(b,'B','Online certification versus conditional forensic work');tidy(b)
    f.text(.18,.02,'Phase contributions partition the PCG-MAS resource envelope.\nThey are not phase-matched differences against NoCert.',ha='left')
    for ax in f.axes:ax.tick_params(labelsize=13)
    r.finish(f,21)
