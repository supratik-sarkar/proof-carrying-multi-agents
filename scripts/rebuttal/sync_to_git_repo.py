#!/usr/bin/env python3
"""Phase 2 Sync Script: Sync audited artifacts/ and scripts/rebuttal/ to Git Repo."""

import os
import shutil
import subprocess
from pathlib import Path

SRC_REPO = Path("/Users/supratiksarkar/Desktop/pcg-neurips2026")
DST_REPO = Path("/Users/supratiksarkar/Desktop/My_Git/proof-carrying-multi-agents")

# 1. Sync artifacts/
src_art = SRC_REPO / "artifacts"
dst_art = DST_REPO / "artifacts"

print("--- SYNCING ARTIFACTS TO GIT REPOSITORY ---")
if dst_art.exists():
    shutil.rmtree(dst_art)

shutil.copytree(src_art, dst_art, ignore=shutil.ignore_patterns(".*", "*.pyc", "__pycache__"))
print(f"Synced {src_art} -> {dst_art}")

# 2. Sync scripts/rebuttal/
src_reb_scripts = SRC_REPO / "scripts" / "rebuttal"
dst_reb_scripts = DST_REPO / "scripts" / "rebuttal"
dst_reb_scripts.parent.mkdir(parents=True, exist_ok=True)

if dst_reb_scripts.exists():
    shutil.rmtree(dst_reb_scripts)

shutil.copytree(src_reb_scripts, dst_reb_scripts, ignore=shutil.ignore_patterns(".*", "*.pyc", "__pycache__"))
print(f"Synced {src_reb_scripts} -> {dst_reb_scripts}")

# 3. Check results/ redundancy
dst_res = DST_REPO / "results"
if dst_res.exists():
    print("Removing redundant results/ directory from Git repository...")
    shutil.rmtree(dst_res)

# 4. Clean .DS_Store
for p in DST_REPO.rglob(".DS_Store"):
    p.unlink()

print("Sync completed cleanly!")
