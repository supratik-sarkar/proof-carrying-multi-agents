"""Data-independent vector artwork for the six protocol schematics.

All geometry, typography, arrows and labels are authored here. No reference value or
experimental result is loaded by this module.
"""
from pathlib import Path
import math,os,subprocess,shutil
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,white
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase.pdfmetrics import stringWidth, registerFont
from reportlab.pdfbase.ttfonts import TTFont
from matplotlib.font_manager import findfont
registerFont(TTFont("FigureSans",findfont("DejaVu Sans")))
registerFont(TTFont("FigureSansBold",findfont("DejaVu Sans:weight=bold")))
from figure_contract import ROOT,NAMES,STATIC
OUT=Path(os.environ.get('PCG_MANUSCRIPT_IMAGES',ROOT/'images'))
INK='#183044';MUTED='#647C8B';BLUE='#2374AB';TEAL='#008978';RED='#C53645';GOLD='#D39522';PURPLE='#7356A6';PALE='#F1F6F9'
W=1100
class Art:
    def __init__(self,n,h,title,subtitle):
        self.n=n;self.h=h*1.55;OUT.mkdir(parents=True,exist_ok=True)
        self.c=canvas.Canvas(str(OUT/(NAMES[n-1]+'.pdf')),pagesize=(W,self.h),invariant=1,initialFontName="FigureSans")
        self.c.setTitle(title);self.c.setAuthor('');self.c.setSubject('Protocol schematic: data independent')
        self.text(25,30,title,25,INK,bold=True);self.text(25,54,subtitle,15.5,MUTED)
    def text(self,x,y,s,size=12,color=INK,bold=False,center=False):
        c=self.c;c.setFillColor(HexColor(color));c.setFont('FigureSansBold' if bold else 'FigureSans',size)
        for i,line in enumerate(s.split('\n')):
            (c.drawCentredString if center else c.drawString)(x,self.h-y*1.55-i*size*1.25,line)
    def box(self,x,y,w,h,title,body='',color=BLUE,fill=PALE,size=13):
        c=self.c;c.setStrokeColor(HexColor(color));c.setFillColor(HexColor(fill));c.setLineWidth(1.2);c.roundRect(x,self.h-(y+h)*1.55,w,h*1.55,10,stroke=1,fill=1)
        c.setFillColor(HexColor(color));c.roundRect(x+12,self.h-y*1.55-10,28,4,2,stroke=0,fill=1)
        lines=simpleSplit(title,'FigureSansBold',16,w-28)
        self.text(x+14,y+26,'\n'.join(lines),16,color,True)
        if body:
            lines_body=[]
            for line in body.split('\n'): lines_body.extend(simpleSplit(line,'FigureSans',15.5,w-28))
            top=y+26+(len(lines)*20+12)/1.55
            self.text(x+14,top,'\n'.join(lines_body),15.5,INK)
    def line(self,x1,y1,x2,y2,color=MUTED,dash=False,arrow=True):
        c=self.c;c.setStrokeColor(HexColor(color));c.setLineWidth(1.7);c.setDash(5,4) if dash else c.setDash();c.line(x1,self.h-y1*1.55,x2,self.h-y2*1.55);c.setDash()
        if arrow:
            angle=math.atan2(y2-y1,x2-x1);s=7;p=c.beginPath();p.moveTo(x2,self.h-y2*1.55)
            for off in [-.5,.5]:p.lineTo(x2-s*math.cos(angle+off),self.h-y2*1.55+s*math.sin(angle+off))
            p.close();c.setFillColor(HexColor(color));c.drawPath(p,fill=1,stroke=0)
    def banner(self,y,s,color=TEAL):
        self.c.setFillColor(HexColor(color));self.c.roundRect(25,self.h-y-33,W-50,33,6,stroke=0,fill=1);self.text(W/2,y+22,s,12,'#FFFFFF',True,True)
    def finish(self):
        self.c.showPage();self.c.save()
        renderer=os.environ.get('PDFTOPPM') or shutil.which('pdftoppm')
        if not renderer: raise RuntimeError('Install Poppler or set PDFTOPPM to its executable.')
        subprocess.run([renderer,'-singlefile','-scale-to','1900','-png',str(OUT/(NAMES[self.n-1]+'.pdf')),str(OUT/NAMES[self.n-1])],check=True,stdout=subprocess.DEVNULL)

def workflow():
    from protocol_figures import workflow as render
    render()

def replay():
    a=Art(6,330,'One committed run, two different questions','Replay failure and environmental drift are distinct audit outcomes.')
    a.box(25,126,192,98,'COMMITTED RUN','Evidence + tool outputs\nHashes + certificate',PURPLE)
    a.box(285,88,255,89,'SNAPSHOT REPLAY','Use the same committed artifacts\nRecompute the acceptance checks',TEAL)
    a.box(285,203,255,89,'FRESH EXECUTION','Reissue retrieval / API / tool calls\nCompare against the snapshot',BLUE)
    a.line(217,162,285,128,TEAL);a.line(217,190,285,243,BLUE)
    a.box(610,88,455,89,'REPRODUCIBLE OR REPLAY FAILURE','Same inputs, same checks: certificate reproduces\nA replay mismatch challenges historical checkability',TEAL)
    a.box(610,203,455,89,'UNCHANGED OR DRIFT ALERT','Changed web page / API version / tool response\nDrift does not by itself invalidate the historical certificate',GOLD)
    a.line(540,132,610,132,TEAL);a.line(540,245,610,245,BLUE);a.finish()

def support():
    a=Art(7,400,'Separated support, then replay-based responsibility','Different-looking branches may still share a hidden dependency.')
    for y,t,b,c in [(88,'SUPPORT PATH A','Root A  /  evidence set A\nRetriever > parser > delegate A',BLUE),(188,'SUPPORT PATH B','Root B  /  evidence set B\nTool > parser > delegate B',TEAL),(288,'CANDIDATE PATH C','Shared root / evidence\nHidden dependency with A',RED)]:a.box(25,y,245,82,t,b,c)
    a.box(345,88,345,160,'SEPARATION GATE','Different provenance roots\nTool overlap ≤ κ\nEvidence overlap ≤ δ\nSemantic overlap ≤ ζ',PURPLE)
    a.line(270,126,345,126,BLUE);a.line(270,226,345,203,TEAL);a.line(270,326,345,282,RED,dash=True)
    a.box(345,269,345,65,'REJECT SHARED DEPENDENCY','Not independent under the declared overlap tests',RED)
    a.box(770,88,305,105,'CLAIM + CERTIFICATE','Accepted support paths are committed\nwith the applicable obligation checks',TEAL);a.line(690,145,770,129,TEAL)
    a.box(770,217,305,113,'MASK AND REPLAY','Mask a committed component\nRecompute the decision\nReport diagnostic responsibility',BLUE);a.line(921,193,921,217,BLUE)
    a.text(350,373,'Separation is checked under declared thresholds; it is not unconditional statistical independence.',11,MUTED);a.finish()

def boundary():
    a=Art(8,240,'What the certificate guarantees, and what remains monitored','A recomputable decision contract is not a guarantee of universal world truth.')
    a.box(60,89,460,112,'RECOMPUTED / CONTRACTUAL','Integrity of committed artifacts\nSnapshot replay against committed state\nPolicy obligations in the declared scope\nSemantic-support checks under committed checker IDs',TEAL)
    a.box(580,89,460,112,'MONITORED / NOT GUARANTEED','Residual source and world-truth error\nUncovered deployment strata\nDistribution shift and changing environments\nFresh-execution drift and diagnostic uncertainty',GOLD)
    a.line(550,83,550,208,PURPLE,arrow=False);a.text(550,227,'DECLARED TRUST BOUNDARY',12,PURPLE,True,True);a.finish()

def search():
    a=Art(9,360,'Certifying search: find a sufficient committed witness','The search proposes evidence; deterministic checking decides whether it certifies the obligations.')
    a.box(25,144,171,95,'OBLIGATION MAP','Atomic claims\nApplicable channels\nDeclared thresholds',BLUE)
    for y,t in [(83,'WINDOW A'),(171,'WINDOW B'),(259,'WINDOW C')]:a.box(247,y,182,70,t,'Evidence + checker score',PURPLE);a.line(196,191,247,y+35,PURPLE)
    a.text(338,66,'FIXED SEARCH BUDGET B',11,PURPLE,True,True)
    a.box(485,139,195,111,'DETERMINISTIC PRUNE','Rank under fixed rules\nDiscard insufficient sets\nTest surviving obligations',BLUE)
    for y in [118,206,294]:a.line(429,y,485,192,BLUE)
    a.box(735,142,157,104,'MINIMAL WITNESS','S*\nSufficient committed\nevidence subset',TEAL);a.line(680,193,735,193,TEAL)
    a.box(937,142,139,140,'CERTIFICATE Z','Witness + hashes\nChecker / model IDs\nConsumer replay',TEAL);a.line(892,193,937,193,TEAL)
    a.text(490,312,'No sufficient witness within budget: fail closed.',12,'#000000',True);a.finish()

def dag():
    from build_theory_dependency import main
    main()

FUNCTIONS={2:workflow,6:replay,7:support,8:boundary,9:search,10:dag}
if __name__=='__main__':
    import sys
    for n in ([int(v) for v in sys.argv[1:]] or sorted(STATIC)):FUNCTIONS[n]()
