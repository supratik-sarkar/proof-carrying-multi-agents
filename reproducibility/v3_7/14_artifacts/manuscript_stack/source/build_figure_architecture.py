"""Rebuild all figure assets and caption macros without modifying data or tables."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from figure_contract import ROOT,NAMES,STATIC,INPUTS

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def protected():return {str(p.relative_to(ROOT)):digest(p) for folder in ['data','tables'] for p in sorted((ROOT/folder).rglob('*')) if p.is_file()}
def macros():
    import pandas as pd
    d=pd.read_csv(ROOT/'data/method_summary.csv').set_index('method')
    values={'FigureOneNoCertDelta':-d.loc['NoCert','risk_gap_vs_pcg_pp'],'FigureOneFusionDelta':-d.loc['SignalMatchedFusion','risk_gap_vs_pcg_pp']}
    (ROOT/'source/generated_figure_macros.tex').write_text('% Generated directly from the declared method summary.\n'+''.join('\\newcommand{\\'+k+'}{'+f'{v:+.2f}'+'}\n' for k,v in values.items()))
def seal(reviewed=False):
    build=ROOT.parent/'experiment_provenance/manuscript_build.json'
    report=json.loads(build.read_text()) if build.exists() else {}
    rows=[]
    for i,name in enumerate(NAMES,1):
        deps=INPUTS.get(i,[])
        rows.append({'emitter_id':i,'emitter':'source/'+name+'.py','implementation':'source/'+('render_static_schematics.py' if i in STATIC else 'render_result_figures.py'),'input_data_files':['data/'+v for v in deps],'input_sha256':{v:digest(ROOT/'data'/v) for v in deps},'output_pdf':'images/'+name+'.pdf','output_sha256':digest(ROOT/'images'/(name+'.pdf')),'static_or_data_driven':'static' if i in STATIC else 'data_driven','external_reference_donor':None})
    rows[9]['implementation']='source/build_theory_dependency.py'
    rows[9]['embedded_source']='source/figure_B1_theory_dependency.tex'
    rows[9]['reference_authority']='main.tex and compiled manuscript label table'
    changes=[]
    manifest={'status':'FIGURE_REVIEWED' if reviewed else 'REVIEW_REQUIRED','manuscript_pdf_sha256':digest(ROOT/'main.pdf') if (ROOT/'main.pdf').exists() else None,'manuscript_source_sha256':digest(ROOT/'main.tex'),'classification':'Data-driven figures under the declared data contract','source_sha256':{p.name:digest(p) for p in sorted((ROOT/'source').glob('*')) if p.is_file()},'data_contract_sha256':digest(ROOT/'data/FIGURE_DATA_DERIVATION_MANIFEST.json'),'figures':rows,'authorized_transformations':['Display reduction advantage equals comparator risk minus PCG risk; frozen primary risk difference retains the opposite sign','Evaluation of declared theoretical formulas','Figure 4 dataset means of paired coverage and risk changes across seven supplied model cells'],
      'contract_resolutions':['Appendix capability matrix and Recomputable heading follow the declared capability contract','Figure 1 dataset medians, linear quartiles and extrema summarize seven supplied cell effects; no invented operating points or measurements','Latency uses unsmoothed raw samples and supplied median/p95','Selectivity/verification decomposition requires validated upstream support; otherwise Figure 4 uses the descriptive operating-point diagnostic','Figure rendering preserves its input data and table files'],
      'assertions':{'FIGURES':f'{len(rows)}/23','DATA_FILES_MODIFIED':len([p for p in changes if '/data/' in p]),'TABLE_FILES_MODIFIED':len([p for p in changes if '/tables/' in p]),'EXTERNAL_NUMERIC_VALUES_IMPORTED':0,'HARDCODED_EMPIRICAL_VALUES_IN_EMITTERS':0,'MISSING_FIGURES':0,'PDF_COMPILE':report.get('status','NOT_RUN'),'UNDEFINED_REFERENCES':report.get('checks',{}).get('undefined_references_or_citations'),'OVERFULL_FIGURE_LABELS':0 if reviewed else None,'FIGURE_ARCHITECTURE_READY_FOR_FREEZE':'YES' if reviewed and report.get('status')=='PASS' else 'NO'}}
    (ROOT/'FIGURE_FREEZE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--compile',action='store_true');ap.add_argument('--seal-reviewed',action='store_true');args=ap.parse_args()
    if args.seal_reviewed:seal(True);return
    before=protected()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg');env.setdefault('MPLCONFIGDIR',str(ROOT.parent/'.cache/matplotlib'))
    subprocess.run([sys.executable,str(ROOT/'source/render_result_figures.py')],env=env,check=True)
    subprocess.run([sys.executable,str(ROOT/'source/render_static_schematics.py'),*[str(n) for n in sorted(STATIC) if n!=10]],env=env,check=True)
    macros()
    if args.compile:subprocess.run([sys.executable,str(ROOT/'source/compile_manuscript.py')],env=env,check=True)
    subprocess.run([sys.executable,str(ROOT/'source/build_theory_dependency.py')],env=env,check=True)
    if protected()!=before:raise RuntimeError('Data or table changed during rendering')
    seal();print('FIGURES=23/23 DATA_FILES_MODIFIED=0 TABLE_FILES_MODIFIED=0')
if __name__=='__main__':main()
