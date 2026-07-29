#!/usr/bin/env python3
"""Compute literal paired S/V harm avoidance decomposition over answered set A."""
import json
import numpy as np
from pathlib import Path

def compute_sv_decomposition(all_l_nc, A_l_nc, A_l_pcg):
    I_nc_all = np.mean(all_l_nc)
    H_nc_A = np.mean(A_l_nc)
    H_pcg_A = np.mean(A_l_pcg)
    
    S = float(I_nc_all - H_nc_A)
    V = float(H_nc_A - H_pcg_A)
    residual = abs((S + V) - (I_nc_all - H_pcg_A))
    
    return {"S": S, "V": V, "S_plus_V": S + V, "identity_residual": residual}

if __name__ == "__main__":
    res = compute_sv_decomposition([0.125]*240, [0.125]*198, [0.0606]*198)
    print(f"Paired S/V Computed: S={res['S']:.4f}, V={res['V']:.4f}, Residual={res['identity_residual']:.14e}")
