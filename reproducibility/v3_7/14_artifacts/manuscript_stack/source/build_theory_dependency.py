"""Export the static reference DAG using the frozen public label table."""
from pathlib import Path
import os,re,shutil,subprocess,tempfile
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
WORK=Path(tempfile.mkdtemp(prefix='pcg-theory-map-'))

def main():
    WORK.mkdir(parents=True,exist_ok=True)
    source=(ROOT/'source/figure_B1_theory_dependency.tex').read_text()
    labels=set(re.findall(r'\\(?:dagref\{|hyperref\[)([^}\]]+)',source))-{'#1'}
    label_source=ROOT/'source/figure_B1_reference_labels.tex'
    label_text=label_source.read_text()
    declared=set(re.findall(r'^\\newlabel\{([^}]+)\}',label_text,re.MULTILINE))
    if not labels<=declared:raise ValueError('Missing frozen labels: '+repr(labels-declared))
    (WORK/'references.tex').write_text(label_text)
    (WORK/'figure_B1_theory_dependency.tex').write_text(source)
    tex=r'''\documentclass[border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times,amsmath,amssymb,tikz,hyperref}
\usetikzlibrary{arrows.meta,positioning}
\hypersetup{hidelinks}
\input{references.tex}
\begin{document}
\renewcommand{\hyperref}[2][]{#2}
\input{figure_B1_theory_dependency.tex}
\end{document}
'''
    (WORK/'dag.tex').write_text(tex)
    env=dict(os.environ,XDG_CACHE_HOME=str(WORK/'cache'),SOURCE_DATE_EPOCH='0',FORCE_SOURCE_DATE='1')
    engine=shutil.which('pdflatex')
    if not engine:raise RuntimeError('A TeX engine is required to render this source artifact.')
    with (WORK/'console.txt').open('w') as log_file:
        subprocess.run([engine,'-interaction=nonstopmode','-halt-on-error','dag.tex'],cwd=WORK,env=env,check=True,stdout=log_file,stderr=subprocess.STDOUT)
    log=(WORK/'dag.log').read_text();assert 'undefined on input' not in log and 'Overfull' not in log, 'DAG layout/reference warning'
    out=Path(os.environ.get('PCG_MANUSCRIPT_IMAGES',ROOT/'images'));out.mkdir(parents=True,exist_ok=True)
    stem='figure_B1_theory_dependency';shutil.copyfile(WORK/'dag.pdf',out/(stem+'.pdf'))
    subprocess.run([os.environ.get('PDFTOPPM') or shutil.which('pdftoppm'),'-singlefile','-scale-to','2200','-png',str(out/(stem+'.pdf')),str(out/stem)],check=True,stdout=subprocess.DEVNULL)
    assert len(PdfReader(out/(stem+'.pdf')).pages)==1
    print('THEORY_DEPENDENCY_MAP=PASS PRIMARY_TILES=9 EMPIRICAL_VALUES_USED=0')

if __name__=='__main__':main()
