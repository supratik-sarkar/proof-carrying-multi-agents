#!/usr/bin/env python3
"""Sync verified rebuttal artifacts and scripts to the Git repository dynamically."""

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GIT_REPO_ROOT = REPO_ROOT.parent / "My_Git" / "proof-carrying-multi-agents"

def sync_directories():
    print("--- SYNCING ARTIFACTS TO GIT REPOSITORY ---")

    src_art = REPO_ROOT / "artifacts"
    dst_art = GIT_REPO_ROOT / "artifacts"

    src_scr = REPO_ROOT / "scripts" / "rebuttal"
    dst_scr = GIT_REPO_ROOT / "scripts" / "rebuttal"

    if dst_art.exists():
        shutil.rmtree(dst_art)
    shutil.copytree(src_art, dst_art, ignore=shutil.ignore_patterns(".*", "*.pyc", "__pycache__"))
    print(f"Synced {src_art} -> {dst_art}")

    if dst_scr.exists():
        shutil.rmtree(dst_scr)
    shutil.copytree(src_scr, dst_scr, ignore=shutil.ignore_patterns(".*", "*.pyc", "__pycache__"))
    print(f"Synced {src_scr} -> {dst_scr}")

    print("Sync completed cleanly!")

if __name__ == "__main__":
    sync_directories()
