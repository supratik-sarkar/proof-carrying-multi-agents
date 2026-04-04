from __future__ import annotations

import subprocess


def main():
    subprocess.run(["python", "scripts/run_eval.py", "--config", "configs/medmcqa.yaml"], check=True)

    try:
        subprocess.run(["python", "scripts/run_eval.py", "--config", "configs/medqa.yaml"], check=True)
    except subprocess.CalledProcessError:
        print("MedQA run failed. Continuing with MedMCQA only.")


if __name__ == "__main__":
    main()
