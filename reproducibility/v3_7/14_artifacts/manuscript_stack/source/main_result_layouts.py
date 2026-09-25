"""Release accounting and adversarial/shift diagnostics from declared inputs."""
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from figure_labels import direct_labels

def require(frame,columns,name):
    missing=set(columns)-set(frame)
    if missing:raise ValueError(name+' missing columns: '+', '.join(sorted(missing)))
    if frame[list(columns)].isna().any().any():raise ValueError(name+' has missing required fields')

def funnel_data(r):
    d=r.read('release_control_funnel.csv')
    require(d,['stage_order','stage','entered_n','passed_n','failed_n','indeterminate_n','cumulative_pass_rate_pct','terminal_state','failure_reason_class'],'release funnel')
    d=d.sort_values('stage_order')
    if d.stage_order.duplicated().any():raise ValueError('Duplicate funnel stage')
    counts=d[['entered_n','passed_n','failed_n','indeterminate_n']].to_numpy(float)
    if not np.isfinite(counts).all() or (counts<0).any() or not np.equal(counts,counts.astype(int)).all():raise ValueError('Invalid funnel counts')
    np.testing.assert_array_equal(d.entered_n,d.passed_n+d.failed_n+d.indeterminate_n)
    np.testing.assert_array_equal(d.entered_n.iloc[1:],d.passed_n.iloc[:-1])
    np.testing.assert_allclose(d.cumulative_pass_rate_pct,100*d.passed_n/d.entered_n.iloc[0],atol=.000051,rtol=0)
    return d

def release_funnel(r):
    d=funnel_data(r)
    with plt.rc_context({'font.size':17}):
        f,ax=plt.subplots(figsize=(18,5.5));f._final_layout=True
        f.subplots_adjust(left=.015,right=.985,top=.98,bottom=.03)
        ax.set_xlim(-.48,len(d)-.52);ax.set_ylim(-1.34,1.15);ax.axis('off')
        for i,row in enumerate(d.itertuples()):
            color=r.TEAL if row.terminal_state=='RELEASED' else r.INK
            if i<len(d)-1:ax.annotate('',xy=(i+.92,0),xytext=(i+.11,0),arrowprops=dict(arrowstyle='->',color=r.INK,lw=1.8))
            ax.scatter(i,0,s=145,color=color,zorder=3)
            ax.text(i,.80,textwrap.fill(row.stage,16),ha='center',va='center',fontsize=20,fontweight='bold',color=color)
            ax.text(i,.29,f'{row.entered_n:,} entered\n{row.passed_n:,} passed',ha='center',va='center',fontsize=20)
            ax.text(i,-.22,f'{row.cumulative_pass_rate_pct:.1f}%',ha='center',fontsize=20,fontweight='bold',color=color)
            if row.failed_n or row.indeterminate_n:
                ax.annotate('',xy=(i,-.64),xytext=(i,-.31),arrowprops=dict(arrowstyle='->',color=r.GRAY,lw=1))
                reason=row.failure_reason_class.replace('_',' ').lower().replace(' or ',' / ')
                detail=f'{row.failed_n:,} failed'
                if row.indeterminate_n:detail+=f' · {row.indeterminate_n:,} indeterminate'
                ax.text(i,-.73,textwrap.fill(detail,18)+'\n'+textwrap.fill(reason,19),ha='center',va='top',fontsize=17,color='#5C6872')
        ax.text(0,-1.29,'Percentages: cumulative retained candidates',ha='left',fontsize=13,color=r.GRAY)
        r.finish(f,22)

def robustness(r):
    inj=r.read('injection_stress.csv');shift=r.read('shift_stress.csv')
    require(inj,['regime','attack_success','accepted_attack','detection','rho_ucb'],'injection stress')
    require(shift,['severity','mode','bound_violation','utility','audit_coverage','alarm_power'],'shift stress')
    if inj.regime.duplicated().any() or shift.duplicated(['severity','mode']).any():raise ValueError('Duplicate stress coordinates')
    if (inj.accepted_attack>inj.attack_success).any():raise ValueError('Accepted attack exceeds upstream success')
    with plt.rc_context({'font.size':17,'axes.titlesize':20,'axes.labelsize':17,'xtick.labelsize':16,'ytick.labelsize':16}):
        f,axes=plt.subplots(2,2,figsize=(16,9.4));f._final_layout=True
        f.subplots_adjust(left=.17,right=.97,bottom=.11,top=.92,wspace=.48,hspace=.52)
        a,b,c,d=axes.flat;y=np.arange(len(inj))
        for yy,row in zip(y,inj.itertuples()):
            a.plot([row.accepted_attack,row.attack_success],[yy,yy],color=r.GRAY,lw=2)
            a.scatter(row.attack_success,yy,facecolors='white',edgecolors=r.INK,s=95,zorder=3)
            a.scatter(row.accepted_attack,yy,color=r.TEAL,edgecolors='white',s=95,zorder=4)
            for v,off in [(row.accepted_attack,-17),(row.attack_success,12)]:a.annotate(f'{v:g}',(v,yy),xytext=(0,off),textcoords='offset points',ha='center',fontsize=17)
        a.set_yticks(y,[textwrap.fill(v,19) for v in inj.regime]);a.set_ylim(len(y)-.45,-.55);a.set_xlim(0,inj.attack_success.max()*1.15)
        a.set_xlabel('Attack rate (%)');r.title(a,'A','Adversarial release leakage')
        a.text(.02,.98,'● Accepted attack     ○ Upstream success',transform=a.transAxes,fontsize=14,color=r.INK,va='top')
        colors=[r.TEAL if 'shared' in v.lower() else '#7F909C' for v in inj.regime]
        b.scatter(inj.rho_ucb,inj.detection,s=100,c=colors,edgecolors='white',linewidths=1.2)
        b.set_xlim(inj.rho_ucb.min()-.3,inj.rho_ucb.max()+.6);b.set_ylim(inj.detection.min()-12,inj.detection.max()+15)
        b.set_xlabel(r'Dependence inflation $\rho_{\mathrm{UCB}}$');b.set_ylabel('Detection (%)');r.title(b,'B','Detection and dependence')
        direct_labels(b,[(row.rho_ucb,row.detection,textwrap.fill(row.regime,18),col) for row,col in zip(inj.itertuples(),colors)],fontsize=17)
        modes=list(shift['mode'].drop_duplicates());palette=['#74838F','#8F8197',r.TEAL]
        if len(modes)>len(palette):raise ValueError('Additional operating modes require an explicit visual contract')
        endpoints_c=[];endpoints_d=[]
        for mode,color in zip(modes,palette):
            g=shift[shift['mode']==mode].sort_values('severity')
            c.plot(g.severity,g.bound_violation,color=color,lw=2,marker='o',ms=3)
            d.plot(g.audit_coverage,g.utility,color=color,lw=1.8)
            d.scatter(g.audit_coverage,g.utility,s=18+g.alarm_power*.9,color=color,alpha=.75,edgecolors='white',linewidths=.5)
            last=g.iloc[-1]
            endpoints_c.append((last.severity,last.bound_violation,textwrap.fill(mode,16),color))
            endpoints_d.append((last.audit_coverage,last.utility,textwrap.fill(mode,16),color))
        c.set_xlim(shift.severity.min()-.02,shift.severity.max()+.72);c.set_ylim(0,shift.bound_violation.max()*1.2)
        c.set_xlabel('Shift severity');c.set_ylabel('Bound violations (%)');r.title(c,'C','Bound validity under shift')
        d.set_xlim(shift.audit_coverage.min()-3,shift.audit_coverage.max()+3);d.set_ylim(shift.utility.min()-6,shift.utility.max()+3)
        d.set_xlabel('Audit coverage (%)');d.set_ylabel('Utility (%)');r.title(d,'D','Utility and monitoring')
        d.text(.01,.98,'Marker area increases with alarm power',transform=d.transAxes,va='top',fontsize=13,color=r.GRAY)
        direct_labels(c,endpoints_c,fontsize=17,right_only=True);direct_labels(d,endpoints_d,fontsize=17)
        for ax in axes.flat:r.tidy(ax)
        r.finish(f,23)
