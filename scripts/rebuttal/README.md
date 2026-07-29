# Rebuttal Server Execution Pipeline

## Quick Start on GPU Server / Colab Pro:

1. **Plan Run:**
   ```bash
   python scripts/rebuttal/plan_56cell_run.py --plan
   ```

2. **Execute 56-Cell Matrix on Server:**
   ```bash
   python scripts/rebuttal/run_56cell_server.py --output-root /path/to/server_runs --resume
   ```

3. **Finalize Rebuttal Artifacts:**
   ```bash
   python scripts/rebuttal/finalize_rebuttal_artifacts.py --server-output-dir /path/to/server_runs
   ```
