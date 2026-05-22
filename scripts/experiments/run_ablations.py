# scripts/v5_ablation_runner.py
from __future__ import annotations

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable-replay", action="store_true")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--disable-responsibility", action="store_true")
    parser.add_argument("--disable-risk-controller", action="store_true")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    return parser.parse_args()


def ablation_name(args) -> str:
    if args.disable_replay:
        return "PCG-MAS:NoReplay"
    if args.k == 1:
        return "PCG-MAS:NoRedundancy"
    if args.disable_responsibility:
        return "PCG-MAS:NoResp"
    if args.disable_risk_controller:
        return "PCG-MAS:NoRiskCtrl"
    return "PCG-MAS"


def main():
    args = parse_args()
    name = ablation_name(args)
    print(f"Running ablation: {name}")

    # Wiring this into our existing PCG runner:
    # replay_enabled = not args.disable_replay
    # redundancy_k = args.k
    # responsibility_enabled = not args.disable_responsibility
    # risk_controller_enabled = not args.disable_risk_controller
    #
    # Then to write per-example outputs with fields:
    # system_name, accepted, harm, token_count, latency, checker_calls, replay_calls


if __name__ == "__main__":
    main()


# \begin{figure}[!t]
#     \centering
#     \includegraphics[width=\linewidth]{\figdir ablations.pdf}
#     \caption{\small \textbf{PCG-MAS ablations under clean and adversarial replay stress.}
#     The left panel reports clean accepted-harm rates; the right panel stresses mutable external
#     state with adversarial evidence/tool perturbation
#     \(\varepsilon_{\mathrm{adv}}=0.25\) and fresh-tool recall probability
#     \(p_{\mathrm{fresh}}=0.30\). Removing replay is most damaging under fresh-mode drift because
#     committed snapshots can no longer isolate the checker from changing tool/API outputs. Removing
#     redundancy and risk control also increases adversarial harm, while removing responsibility mainly
#     affects diagnosis rather than first-order acceptance. Lower is better.}
#     \label{fig:ablations}
# \end{figure}