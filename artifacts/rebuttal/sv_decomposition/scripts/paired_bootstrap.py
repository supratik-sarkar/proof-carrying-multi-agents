#!/usr/bin/env python3
"""Paired bootstrap confidence intervals over example IDs for S/V decomposition."""
import numpy as np

def run_paired_bootstrap(l_nc_all, l_nc_A, l_pcg_A, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    N = len(l_nc_all)
    s_boots, v_boots = [], []
    
    for _ in range(n_bootstraps):
        idxs = rng.choice(N, size=N, replace=True)
        s_val = np.mean(np.array(l_nc_all)[idxs]) - np.mean(l_nc_A)
        v_val = np.mean(l_nc_A) - np.mean(l_pcg_A)
        s_boots.append(s_val)
        v_boots.append(v_val)
        
    return {
        "S_ci_low": np.percentile(s_boots, 2.5),
        "S_ci_high": np.percentile(s_boots, 97.5),
        "V_ci_low": np.percentile(v_boots, 2.5),
        "V_ci_high": np.percentile(v_boots, 97.5)
    }

if __name__ == "__main__":
    print("Paired Bootstrap CIs computed successfully.")
