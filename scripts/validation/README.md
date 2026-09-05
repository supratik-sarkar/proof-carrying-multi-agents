# Validation Server Execution Pipeline

## Quick Start on GPU Server / Colab Pro:

1. **Plan Run:**
   ```bash
   python scripts/validation/plan_56cell_run.py --plan
   ```

2. **Execute 56-Cell Matrix on Server:**
   ```bash
   python scripts/validation/run_56cell_server.py --output-root /path/to/server_runs --resume
   ```

3. **Finalize Validation Artifacts:**
   ```bash
   python scripts/validation/finalize_artifacts.py --server-output-dir /path/to/server_runs
   ```
