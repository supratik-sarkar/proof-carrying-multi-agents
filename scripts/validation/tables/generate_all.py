#!/usr/bin/env python3
"""Master entry point for generating all synthetic placeholder tables and running validation."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_records
import render_manuscript_tables
import render_benchmark_tables
import validate_all_tables

def main():
    print("=================================================================")
    print("=== SYNTHETIC PLACEHOLDER TABLE GENERATION PIPELINE (56 CELLS) ===")
    print("=================================================================")

    # Step 1: Generate per-example synthetic records
    records_file = generate_records.generate()

    # Step 2: Render manuscript tables (Tables 1-18)
    render_manuscript_tables.render_all(records_file)

    # Step 3: Render validation tables in eight subdirectories
    render_benchmark_tables.render_validation_placeholder_tables()

    # Step 4: Validate canonical consistency & run negative tests
    if not validate_all_tables.validate():
        print("[ERROR] Table validation failed!")
        sys.exit(1)

    print("=================================================================")
    print("[SUCCESS] ALL SYNTHETIC PLACEHOLDER TABLES GENERATED & VERIFIED!")
    print("=================================================================")
    return 0

if __name__ == "__main__":
    sys.exit(main())
