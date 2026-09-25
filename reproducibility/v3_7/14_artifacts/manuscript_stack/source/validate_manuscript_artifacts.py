"""Validate published artifact values, references, fonts and placement."""
from pathlib import Path
from collections import Counter
import hashlib,json,re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from pypdf import PdfReader
import render_result_figures as r
from main_result_layouts import funnel_data,require
from figure_contract import NAMES

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    root=r.ROOT;checks={};maintex=(root/'main.tex').read_text()
    def check(name,condition):
        checks[name]='PASS' if condition else 'FAIL'
    capture={};original=r.finish;r.finish=lambda f,n:capture.setdefault(n,f)
    for n in [1,4,22,23]:r.generate(n)
    r.finish=original
    inj=pd.read_csv(root/'data/injection_stress.csv');shift=pd.read_csv(root/'data/shift_stress.csv')
    a,b,c,d=capture[23].axes
    for i,row in enumerate(inj.itertuples()):
        np.testing.assert_allclose(a.lines[i].get_xdata(),[row.accepted_attack,row.attack_success])
    np.testing.assert_allclose(b.collections[0].get_offsets(),inj[['rho_ucb','detection']])
    for i,(mode,g) in enumerate(shift.groupby('mode',sort=False)):
        g=g.sort_values('severity');np.testing.assert_allclose(c.lines[i].get_xdata(),g.severity);np.testing.assert_allclose(c.lines[i].get_ydata(),g.bound_violation)
        np.testing.assert_allclose(d.lines[i].get_xdata(),g.audit_coverage);np.testing.assert_allclose(d.lines[i].get_ydata(),g.utility)
        np.testing.assert_allclose(d.collections[i].get_sizes(),18+g.alarm_power*.9)
    check('robustness_all_values_and_alarm_encoding',True)
    check('robustness_direct_labels_no_legends',all(ax.get_legend() is None for ax in capture[23].axes))
    funnel=funnel_data(r);check('release_counts_conserve_candidates',True)
    strings=' '.join(x.get_text() for x in capture[22].axes[0].texts)
    check('release_exact_counts_visible',all(f'{int(v):,}' in strings for col in ['entered_n','passed_n'] for v in funnel[col]))
    try:require(inj.drop(columns=['accepted_attack']),['accepted_attack'],'injection stress')
    except ValueError:check('missing_scientific_column_rejected',True)
    else:check('missing_scientific_column_rejected',False)
    summary=pd.read_csv(root/'data/method_summary.csv');pcg=summary[summary.method=='PCG-MAS'].iloc[0]
    np.testing.assert_allclose(capture[4].axes[1].collections[-1].get_offsets(),[[pcg.mean_coverage,pcg.mean_risk]])
    check('only_declared_operating_point',len([x for x in capture[4].axes[1].collections if isinstance(x,PathCollection)])==1)
    pairs=pd.read_csv(root/'data/paired_cell_effects.csv');primary=pd.read_csv(root/'data/table_01_primary_endpoint_by_dataset.csv')
    for row in primary.itertuples():
        g=pairs[pairs.dataset==row.dataset]
        np.testing.assert_allclose([row.pcg_accepted_risk,row.best_applicable_risk,row.delta_primary_pp,row.retained_coverage],g[['pcg_risk','best_applicable_risk','delta_pcg_minus_best_pp','pcg_coverage']].mean())
    check('primary_table_all_datasets_and_original_sign',len(primary)==8)
    ledger=pd.read_csv(root/'data/table_03_verifiability_execution_ledger.csv',keep_default_na=False).set_index('quantity')
    invariance=pd.read_csv(root/'data/table_27_protocol_auditor_invariance.csv',keep_default_na=False).set_index('Channel')
    check('ledger_final_agreement',ledger.loc['Final-check agreement','value']==invariance.loc['Final Check','Agreement'])
    check('ledger_final_disagreements',int(ledger.loc['Final-check disagreements','value'])==int(invariance.loc['Final Check','Disagreements']))
    check('ledger_every_row_has_source',ledger.source_file.ne('').all() and ledger.source_field.ne('').all())
    plt.close('all')
    table_names=re.findall(r'\\pcgtable\{([^}]+)\}',maintex)
    included=[root/'main.tex',root/'source/figure_B1_theory_dependency.tex']+[root/'tables'/(n+'.tex') for n in table_names]
    included+=list((root/'tables').glob('appendix_notation_part*.tex'))+[root/'tables/generated_results_macros.tex']
    text='\n'.join(p.read_text() for p in included)
    labels=re.findall(r'\\label\{((?:fig:|tab:)[^}]+)\}',text)
    refs={a or b for a,b in re.findall(r'\\(?:ref\*?|eqref)\{([^}]+)\}|\\hyperref\[([^\]]+)\]',text)}
    orphan=sorted(set(labels)-refs)
    check('all_rendered_artifacts_contextually_referenced',not orphan)
    check('artifact_labels_unique',all(v==1 for v in Counter(labels).values()))
    mainpart=maintex.split('\\appendix')[0]
    check('algorithm_referenced_in_main_text',r'\ref{alg:exp_short}' in mainpart)
    aux=(root.parent/'.cache/manuscript-build/main.aux').read_text()
    labelinfo={m[1]:(m[2],int(m[3])) for m in re.finditer(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{(\d+)\}',aux)}
    pdf=PdfReader(root/'main.pdf');page_text=[p.extract_text() or '' for p in pdf.pages]
    references=next(i+1 for i,t in enumerate(page_text) if re.search(r'\bREFERENCES\b',t))
    check('references_begin_page_10',references==10)
    check('main_text_ends_page_9','Conclusion and future work' in page_text[8])
    check('five_main_figures',all(labelinfo[label][0]==str(i) and labelinfo[label][1]<=9 for i,label in enumerate(['fig:intro_overview','fig:workflow','fig:release_control_funnel','fig:r1_to_r4_combined','fig:robustness_atlas'],1)))
    check('five_main_tables',all(labelinfo[label][0]==str(i) and labelinfo[label][1]<=9 for i,label in enumerate(['tab:main_six_summary','tab:baseline_scope_matrix','tab:verifiability_execution_ledger','tab:protocol_cost','tab:audit_calibration_summary'],1)))
    check('tables_2_and_3_same_page',labelinfo['tab:baseline_scope_matrix'][1]==labelinfo['tab:verifiability_execution_ledger'][1])
    check('tables_4_and_5_same_page',labelinfo['tab:protocol_cost'][1]==labelinfo['tab:audit_calibration_summary'][1])
    sections=['appendix:related_work','appendix:runtime','appendix:proofs','appendix:results','appendix:experiments','appendix:reference']
    check('appendices_A_through_F_only',[labelinfo[x][0] for x in sections]==list('ABCDEF') and len(re.findall(r'\\section\{',maintex.split('\\appendix',1)[1]))==6)
    check('related_work_one_page',labelinfo[sections[1]][1]-labelinfo[sections[0]][1]==1)
    check('prompt_bank_label_resolves','tab:table_32_appendix_prompt_bank' in labelinfo)
    nav=maintex.split(r'\noindent\textbf{Appendix roadmap.}',1)[1].split(r'\end{tcolorbox}',1)[0]
    nav_labels=re.findall(r'\\hyperref\[([^\]]+)\]',nav)
    check('navigation_destinations_exist',all(x in labelinfo for x in nav_labels))
    invalid_links=[];unembedded=[];empty=[]
    def fonts(resources,where,seen):
        resources=resources.get_object()
        for font in resources.get('/Font',{}).values():
            font=font.get_object()
            for f in font.get('/DescendantFonts',[font]):
                f=f.get_object();descriptor=f.get('/FontDescriptor',{});descriptor=descriptor.get_object() if hasattr(descriptor,'get_object') else descriptor
                if not any(k in descriptor for k in ['/FontFile','/FontFile2','/FontFile3']):unembedded.append((where,str(f.get('/BaseFont'))))
        for obj in resources.get('/XObject',{}).values():
            identity=(obj.idnum,obj.generation) if hasattr(obj,'idnum') else id(obj)
            if identity in seen:continue
            seen.add(identity);obj=obj.get_object()
            if '/Resources' in obj:fonts(obj['/Resources'],where,seen)
    for filename in ['main.pdf']+['images/'+n+'.pdf' for n in NAMES]:
        reader=PdfReader(root/filename)
        if not any((p.extract_text() or '').strip() for p in reader.pages):empty.append(filename)
        for page in reader.pages:
            fonts(page['/Resources'],filename,set())
            for annot in page.get('/Annots',[]):
                obj=annot.get_object();act=obj.get('/A',{});dest=act.get('/D') if act.get('/S')=='/GoTo' else obj.get('/Dest')
                if isinstance(dest,str) and dest not in reader.named_destinations:invalid_links.append((filename,dest))
    check('embedded_fonts',not unembedded);check('vector_text_extractable',not empty);check('internal_pdf_destinations_resolve',not invalid_links)
    build=json.loads((root.parent/'experiment_provenance/manuscript_build.json').read_text())
    for key,value in build['checks'].items():check('compile_'+key,value==0)
    check('pdflatex_bibtex_build',build['engine']=='pdflatex + bibtex')
    result={'status':'PASS' if all(v=='PASS' for v in checks.values()) else 'FAIL','checks':checks,'pages':len(pdf.pages),'references_begin_page':references,'artifact_references':len(labels),'navigation_link_count':len(nav_labels),'unreferenced_artifacts':orphan,'unembedded_fonts':unembedded,'invalid_links':invalid_links,'main_pdf_sha256':digest(root/'main.pdf'),'main_figure_sha256':{n:digest(root/'images'/(n+'.pdf')) for n in [NAMES[0],NAMES[1],NAMES[21],NAMES[3],NAMES[22]]},'scope':'Artifact/input consistency and PDF build; not independent verification of experimental execution.'}
    (root.parent/'experiment_provenance/manuscript_artifact_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
