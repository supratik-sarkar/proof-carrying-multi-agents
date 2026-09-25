"""Compile the manuscript and reject unresolved references or geometric overflow.

Uses Tectonic when available; otherwise falls back to pdflatex + bibtex/bibtex8,
which mirrors a conventional Overleaf-style BibTeX build.
"""
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent/'.cache/manuscript-build'


def run(cmd, *, cwd, env, log_path, append=False):
    mode='a' if append else 'w'
    with log_path.open(mode) as log:
        subprocess.run(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def build(env):
    tectonic=shutil.which('tectonic')
    if not tectonic and Path('/opt/homebrew/bin/tectonic').exists():
        tectonic='/opt/homebrew/bin/tectonic'
    console=WORK/'console.txt'
    if tectonic and os.environ.get('PCG_TEX_ENGINE')=='tectonic':
        run([tectonic,'--keep-logs','--keep-intermediates','--outdir',str(WORK),'main.tex'],cwd=ROOT,env=env,log_path=console)
        return 'tectonic'

    local_bin=ROOT.parent/'.cache/artifact-rebuild/TinyTeX/bin/universal-darwin'
    if local_bin.exists():os.environ['PATH']=str(local_bin)+os.pathsep+os.environ['PATH'];env=dict(env,PATH=os.environ['PATH'])
    pdflatex=shutil.which('pdflatex')
    bibtex=shutil.which('bibtex') or shutil.which('bibtex8')
    if not pdflatex or not bibtex:
        raise SystemExit('Need tectonic, or pdflatex plus bibtex/bibtex8, to compile the manuscript.')
    # Kpathsea uses an empty path component to retain its normal search path.
    env=dict(env)
    env['BIBINPUTS']=str(ROOT)+os.pathsep+env.get('BIBINPUTS','')
    env['BSTINPUTS']=str(ROOT)+os.pathsep+env.get('BSTINPUTS','')
    common=[pdflatex,'-interaction=nonstopmode','-halt-on-error','-output-directory',str(WORK),'main.tex']
    run(common,cwd=ROOT,env=env,log_path=console)
    run([bibtex,'main'],cwd=WORK,env=env,log_path=console,append=True)
    run(common,cwd=ROOT,env=env,log_path=console,append=True)
    run(common,cwd=ROOT,env=env,log_path=console,append=True)
    return 'pdflatex + '+Path(bibtex).name


def main():
    if WORK.exists(): shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)
    env=dict(os.environ, XDG_CACHE_HOME=str(ROOT.parent/'.cache'))
    engine=build(env)
    log=(WORK/'main.log').read_text(errors='replace')
    checks={
        'undefined_references_or_citations':len(re.findall(r'(?:Reference|Citation)[^\n]*undefined|There were undefined',log)),
        'lost_floats':len(re.findall(r'Float\(s\) lost',log)),
        'duplicate_labels':len(re.findall(r'multiply defined',log)),
        'overfull_boxes':len(re.findall(r'Overfull \\[hv]box',log)),
        'oversized_floats':len(re.findall(r'Float too large|Too many unprocessed floats',log)),
    }
    reader=PdfReader(WORK/'main.pdf')
    outside=[]
    for i,page in enumerate(reader.pages,1):
        box=page.mediabox
        for entry in page.get('/Annots',[]):
            rect=entry.get_object().get('/Rect')
            if rect and (rect[0]<box.left-1 or rect[1]<box.bottom-1 or rect[2]>box.right+1 or rect[3]>box.top+1): outside.append(i)
    checks['out_of_page_annotations']=len(outside)
    aux=(WORK/'main.aux').read_text(errors='replace')
    figures=[{'label':m.group(1),'number':int(m.group(2)),'page':int(m.group(3))}
             for m in re.finditer(r'\\newlabel\{(fig:[^}]+)\}\{\{(\d+)\}\{(\d+)\}',aux)]
    figure_numbers=sorted({record['number'] for record in figures})
    checks['figure_numbering_errors']=0 if figure_numbers==list(range(1,len(figure_numbers)+1)) else 1
    report={'status':'FAIL' if any(checks.values()) else 'PASS','engine':engine,'pages':len(reader.pages),'checks':checks,'figures':figures,
            'scope':'Compilation, references, figure numbering and geometric overflow; not scientific claim validation'}
    prov=ROOT.parent/'experiment_provenance'
    prov.mkdir(parents=True,exist_ok=True)
    (prov/'manuscript_build.json').write_text(json.dumps(report,indent=2)+'\n')
    if any(checks.values()): raise SystemExit('MANUSCRIPT_BUILD=FAIL; see experiment_provenance/manuscript_build.json')
    shutil.copyfile(WORK/'main.pdf',ROOT/'main.pdf')
    print(f"MANUSCRIPT_BUILD=PASS ENGINE={engine} PAGES={len(reader.pages)} FIGURES={len(figures)}")

if __name__=='__main__': main()
