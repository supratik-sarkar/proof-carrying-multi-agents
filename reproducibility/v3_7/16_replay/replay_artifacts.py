"""Rebuild stored-data artifacts in isolation; no experiments or raw metric replay."""
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
base=Path(__file__).resolve().parents[1]
stack=base/'14_artifacts/manuscript_stack'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='artifact-replay-') as tmp:
    output=Path(tmp); images=output/'images';tables=output/'tables'
    images.mkdir();tables.mkdir()
    env=dict(os.environ,PCG_MANUSCRIPT_IMAGES=str(images),PCG_MANUSCRIPT_TABLES=str(tables),PYTHONDONTWRITEBYTECODE='1')
    for script in ['render_result_figures.py','render_static_schematics.py','figure_06_backend_invariance_atlas.py','generate_tables.py']:
        subprocess.run([sys.executable,str(stack/'source'/script)],env=env,check=True)
    expected=json.loads((base/'16_replay/expected_outputs.json').read_text())
    for directory,count in [(images,expected['figures']*2),(tables,expected['tables'])]:
        outputs=[p for p in directory.iterdir() if p.is_file()]
        assert len(outputs)==count,(directory.name,len(outputs),count)
        for p in outputs: assert digest(p)==digest(stack/directory.name/p.name),p.name
    print(f"PASS: {expected['figures']} figure pairs and {expected['tables']} tables match stored output bytes.")
