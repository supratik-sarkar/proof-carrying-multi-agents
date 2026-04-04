from __future__ import annotations

import subprocess
from pathlib import Path


MODES = [
    "pcg_full",
    "baseline_selective",
    "baseline_multiagent_no_cert",
    "baseline_lightweight_citation",
    "baseline_posthoc_verify",
]


CONFIG_TEMPLATE = """project_name: proof_carrying_multi_agents
output_dir: outputs

runtime:
  seed: 42
  max_examples: 10
  save_jsonl: true

model:
  backbone_name: "google/flan-t5-base"
  device: "auto"
  max_input_length: 1024
  max_new_tokens: 64
  do_sample: false
  temperature: 0.0

retrieval:
  top_k: 4
  bm25_top_k: 4
  dense_top_k: 4

experiment:
  mode: "{mode}"
  dataset_name: "medmcqa"
  accepted_risk_threshold: 0.35

logging:
  include_text: false
  write_tables: true
  write_figures: true
"""


def main():
    Path("configs").mkdir(exist_ok=True)

    for mode in MODES:
        cfg_path = f"configs/tmp_{mode}.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(CONFIG_TEMPLATE.format(mode=mode))

        subprocess.run(["python", "scripts/run_eval.py", "--config", cfg_path], check=True)


if __name__ == "__main__":
    main()
