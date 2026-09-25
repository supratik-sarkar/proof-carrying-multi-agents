"""Deterministic labels positioned against measured text and point bounds."""
import numpy as np
from matplotlib.transforms import Bbox

MODEL_SHORT = {'GPT-OSS 120B':'GPT-OSS', 'Qwen 3.8 27B':'Qwen',
 'Gemini 3.1 Flash-Lite':'Gemini', 'Llama 4 Scout':'Llama',
 'Nemotron 3 120B-A12B':'Nemotron', 'Gemma 4 26B-A4B':'Gemma',
 'GLM-4.7 Flash':'GLM'}

def direct_labels(ax, points, fontsize=18,right_only=False):
    """Place one leader per named aggregate; return labels for validation."""
    fig=ax.figure; fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    bounds=ax.get_window_extent().padded(-5)
    occupied=[]; result=[]
    coordinates=[ax.transData.transform((x,y)) for x,y,_,_ in points]
    obstacles=[Bbox.from_bounds(x-8,y-8,16,16) for x,y in coordinates]
    for x,y,label,color in points:
        anchor=ax.transData.transform((x,y))
        txt=ax.text(x,y,label,fontsize=fontsize,color=color,va='center',zorder=10,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.88,pad=.8))
        candidates=[]
        for radius in [22,38,58,85,115,150,195,240]:
            for angle in np.linspace(0,2*np.pi,20,endpoint=False):
                if right_only and np.cos(angle)<.2:continue
                pos=anchor+radius*np.array([np.cos(angle),np.sin(angle)])
                txt.set_ha('left' if np.cos(angle)>=0 else 'right')
                txt.set_position(ax.transData.inverted().transform(pos))
                bb=txt.get_window_extent(renderer).padded(4)
                if not (bounds.contains(bb.x0,bb.y0) and bounds.contains(bb.x1,bb.y1)):continue
                collisions=sum(bb.overlaps(b) for b in occupied+obstacles)
                candidates.append((collisions,radius,tuple(pos),txt.get_ha(),bb))
            if candidates and min(v[0] for v in candidates)==0:break
        if not candidates:raise ValueError('No in-panel label placement for '+label)
        collisions,_,pos,ha,bb=min(candidates,key=lambda v:(v[0],v[1]))
        if collisions:raise ValueError('Label overlap requires more panel space: '+label)
        txt.set_ha(ha);txt.set_position(ax.transData.inverted().transform(pos));occupied.append(bb)
        ax.annotate('',xy=(x,y),xytext=txt.get_position(),
                    arrowprops=dict(arrowstyle='-',color=color,lw=.9,shrinkA=4,shrinkB=7),zorder=8)
        result.append(txt)
    return result
