from pathlib import Path
import pandas as pd,re,json
import os
ROOT=Path(__file__).resolve().parents[1];DATA=Path(os.environ.get('PCG_MANUSCRIPT_DATA',ROOT/'data'));OUT=Path(os.environ.get('PCG_MANUSCRIPT_TABLES',ROOT/'tables'));OUT.mkdir(parents=True,exist_ok=True)
def esc(x):
 for plain,math in {'G_t':r'$G_t$','V_entail':r'$V_\vdash$','V_Gamma':r'$V_\Gamma$','V_Pi':r'$V_\Pi$','V_H':r'$V_H$'}.items():
  if plain in str(x) and '$' not in str(x):x=str(x).replace(plain,math)
 # Preserve declared mathematical cells.
 mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
 parts=re.split(r'(\$[^$]*\$)',str(x))
 value=''.join(part if part.startswith('$') and part.endswith('$') else ''.join(mapping.get(c,c) for c in part) for part in parts)
 return re.sub(r'(?<=[a-z])(?=[A-Z])',lambda _: r'\allowbreak{}',value)
CONTRACT=json.loads((ROOT/'source/table_contract.json').read_text())
from prepare_main_tables import RENAMES,build
for p in sorted(DATA.glob('table_*.csv')):
 name=p.stem
 if name in RENAMES or name in RENAMES.values():continue
 df=pd.read_csv(p,dtype=str).fillna('--')
 cols=list(df.columns);n=len(cols);cap=CONTRACT[name]['caption'];lab=CONTRACT[name]['label']
 fs='\\scriptsize' if n>=5 else '\\footnotesize'
 header=' & '.join(esc(c) for c in cols)+r' \\'
 body=[' & '.join(esc(v) for v in r)+r' \\' for r in df.itertuples(index=False,name=None)]
 if len(df)>24:
  # Widths follow column content, with page-breaking rows and repeated headers.
  weights=[]
  for col in cols:
   width=max([len(col)]+[len(str(v)) for v in df[col]])
   weights.append(min(max(width,8),30))
  total=sum(weights)
  align='@{}'+''.join(r'>{\raggedright\arraybackslash}p{'+f'{w/total:.5f}'+r'\dimexpr\linewidth-'+str(2*(n-1))+r'\tabcolsep\relax}' for w in weights)+'@{}'
  lines=[r'\begingroup',fs,r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.12}',r'\begin{longtable}{'+align+'}',r'\caption{'+cap+'}'+r'\label{'+lab+r'}\\',r'\toprule',header,r'\midrule',r'\endfirsthead',r'\multicolumn{'+str(n)+r'}{l}{\footnotesize Continued from previous page} \\',r'\toprule',header,r'\midrule',r'\endhead',r'\midrule',r'\multicolumn{'+str(n)+r'}{r}{\footnotesize Continued on next page} \\',r'\endfoot',r'\bottomrule',r'\endlastfoot']+body+[r'\end{longtable}',r'\endgroup','']
 else:
  align='l'+('c'*(n-1))
  if name in {'table_02_cost_overhead_main','table_03_audit_calibration_summary'}:
   width='0.385\\textwidth' if name=='table_02_cost_overhead_main' else '0.585\\textwidth'
   lines=[r'\begin{minipage}[t]{'+width+'}',r'\centering',fs,r'\captionsetup{type=table}',r'\captionof{table}{'+cap+'}',r'\label{'+lab+'}',r'\vspace{2pt}',r'\begin{adjustbox}{max width=\linewidth}',r'\begin{tabular}{'+align+'}',r'\toprule',header,r'\midrule']+body+[r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{minipage}','']
  else:
   lines=[r'\begin{table}[tbp]',r'\centering',fs,r'\caption{'+cap+'}',r'\label{'+lab+'}',r'\begin{adjustbox}{max width=\linewidth}',r'\begin{tabular}{'+align+'}',r'\toprule',header,r'\midrule']+body+[r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{table}','']
 (OUT/f'{name}.tex').write_text('\n'.join(lines),encoding='utf-8')

build(esc)
