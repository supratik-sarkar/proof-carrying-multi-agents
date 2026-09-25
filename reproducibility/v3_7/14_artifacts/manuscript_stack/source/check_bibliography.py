"""Static bibliography hygiene checks for the manuscript bibliography."""
from pathlib import Path
import re, sys
ROOT=Path(__file__).resolve().parents[1]
BIB=ROOT/'references.bib'
TEX=list(ROOT.rglob('*.tex'))
text=BIB.read_text(encoding='utf-8')

# Parse top-level BibTeX entries with brace depth (sufficient for this repository).
entries=[]
for m in re.finditer(r'@(\w+)\{([^,]+),', text):
    typ,key=m.group(1).lower(),m.group(2).strip(); i=m.end(); depth=1
    while i<len(text) and depth:
        if text[i]=='{': depth+=1
        elif text[i]=='}': depth-=1
        i+=1
    entries.append((typ,key,text[m.start():i]))

def field(block,name):
    m=re.search(r'(?im)^\s*'+re.escape(name)+r'\s*=\s*\{',block)
    if not m: return ''
    i=m.end(); depth=1; start=i
    while i<len(block) and depth:
        if block[i]=='{': depth+=1
        elif block[i]=='}': depth-=1
        i+=1
    return block[start:i-1].strip()

errors=[]; warnings=[]
keys=[k for _,k,_ in entries]
if len(keys)!=len(set(keys)): errors.append('duplicate BibTeX keys')
required={'title','author','year'}
seen_titles={}; seen_dois={}
for typ,key,block in entries:
    vals={f:field(block,f) for f in ['title','author','year','doi','url','journal','booktitle','publisher','pages','volume']}
    for req in required:
        if not vals[req]: errors.append(f'{key}: missing {req}')
    if typ=='inproceedings' and not vals['booktitle']: errors.append(f'{key}: inproceedings missing booktitle')
    if typ=='article' and not vals['journal']: errors.append(f'{key}: article missing journal')
    if re.search(r'(?i)\b(todo|tbd|placeholder|unknown)\b',block): errors.append(f'{key}: placeholder token')
    title_norm=re.sub(r'[^a-z0-9]+','',re.sub(r'[{}]','',vals['title']).lower())
    if title_norm:
        if title_norm in seen_titles: errors.append(f'{key}: duplicate normalized title with {seen_titles[title_norm]}')
        seen_titles[title_norm]=key
    doi=vals['doi'].lower().strip()
    if doi:
        if not doi.startswith('10.'): errors.append(f'{key}: malformed DOI {doi}')
        if doi in seen_dois: errors.append(f'{key}: duplicate DOI with {seen_dois[doi]}')
        seen_dois[doi]=key
    if 'arXiv preprint arXiv:' in vals['journal'] and not vals['url']: errors.append(f'{key}: arXiv entry missing URL')

alltex='\n'.join(p.read_text(encoding='utf-8',errors='ignore') for p in TEX)
cited=set()
for m in re.finditer(r'\\cite(?:p|t|author|year)?(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^}]+)\}',alltex):
    cited.update(x.strip() for x in m.group(1).split(',') if x.strip() and x.strip()!='key')
missing=cited-set(keys); orphan=set(keys)-cited
if missing: errors.append('cited keys missing from bibliography: '+', '.join(sorted(missing)))
if orphan: errors.append('uncited/orphan bibliography entries: '+', '.join(sorted(orphan)))
print(f'BIB_ENTRIES={len(entries)} CITED_KEYS={len(cited)}')
for w in warnings: print('WARN',w)
for e in errors: print('ERROR',e)
if errors: sys.exit(1)
print('BIB_HYGIENE=PASS')
