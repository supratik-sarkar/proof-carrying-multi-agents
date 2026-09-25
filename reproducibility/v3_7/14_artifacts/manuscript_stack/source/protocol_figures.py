"""Vector rendering of runtime objects and the consumer trust boundary."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc

def workflow():
    import render_result_figures as r
    with plt.rc_context({'font.size':18}):
        f,ax=plt.subplots(figsize=(17,5.8));f._final_layout=True
        f.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99)
        ax.set_xlim(0,17);ax.set_ylim(0,5.8);ax.axis('off')
        def txt(x,y,s,size=18,color=r.INK,bold=False):
            ax.text(x,y,s,ha='center',va='center',fontsize=size,color=color,fontweight='bold' if bold else 'normal')
        def arrow(x1,y1,x2,y2,color=r.GRAY,dashed=False):
            ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=1.3,color=color,linestyle='--' if dashed else '-'))
        ax.add_patch(Circle((.65,4.5),.17,fill=False,lw=1.7,color=r.INK))
        ax.add_patch(Arc((.65,3.94),.78,.64,theta1=0,theta2=180,lw=1.7,color=r.INK))
        txt(.65,3.25,'Human\nrequest',17);txt(.65,2.55,'Multimodal',14)
        ax.add_patch(Rectangle((1.55,2.5),2.45,2.65,facecolor='#F2F4F6',edgecolor='none'))
        txt(2.77,4.82,'Arbitrary runtime',18,bold=True)
        txt(2.77,3.85,'LLM(s)\nRetrieval · tools\nMemory · delegation',16)
        txt(2.77,2.84,r'Candidate $c$ · graph $G_t$',16)
        arrow(1.02,4.05,1.5,4.05);arrow(4.02,4.05,4.55,4.05)
        ax.plot([4.32,4.32],[1.65,5.12],ls='--',lw=1.1,color=r.GRAY)
        txt(9.7,5.53,'PCG-MAS  ·  model-, dataset-, and agent-stack independent',21,r.TEAL,True)
        centers=[5.85,8.8,12.05]
        for x in centers:ax.plot([x-1.12,x+1.12],[4.52,4.52],color=r.TEAL,lw=2)
        txt(5.85,4.85,'Commitment record',18,bold=True)
        txt(5.85,3.71,'Request · evidence $S$\nTool outputs · hashes\nPolicy / checker IDs',16)
        txt(8.8,4.85,'Unified certificate',18,bold=True)
        txt(8.8,3.82,r'$Z=(c,S,\Pi,\Gamma,$'+'\n'+r'$\mathcal{A},\chi,\mathrm{meta})$',20)
        txt(8.8,2.87,'Pinned, closed witness',16)
        txt(12.05,4.85,'Independent checker',18,bold=True)
        txt(12.05,3.83,r'$\mathrm{Check}(Z;G_t)$'+'\n'+r'$=V_HV_\Pi V_\Gamma V_\vdash$',20)
        txt(12.05,2.87,'Applicable obligations pass',15)
        arrow(7.04,4.03,7.53,4.03);arrow(10.10,4.03,10.55,4.03)
        txt(8.35,1.87,r'$(\delta,\kappa,\zeta)$-separated support',18)
        txt(8.35,1.3,r'Quorum $A_t^{(k,q)}(c)$',18)
        arrow(12.05,2.56,12.05,2.05,r.TEAL)
        txt(12.05,1.78,'Certificate-derived\nrisk / control policy',17,r.TEAL,True)
        arrow(10.18,1.65,10.65,1.65,r.TEAL)
        arrow(13.38,1.7,14.18,1.7,r.TEAL)
        txt(15.45,2.35,'Controlled release',18,r.TEAL,True)
        txt(15.45,1.63,'Answer · Verify\nEscalate · Refuse',18)
        ax.plot([4.7,16.55],[.89,.89],color='#CBD4D9',lw=.9)
        txt(10.6,.50,'Audit sidecar: replay · drift · coverage gap · responsibility / mask replay · operations',15,r.GRAY)
        txt(2.45,.85,'Trust boundary:\nconsumer recomputes\nthe declared decision',15,r.GRAY)
        r.finish(f,2)
