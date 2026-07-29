#!/usr/bin/env python3
"""Canonical Metric Calculation Functions for PCG-MAS Rebuttal Pipeline."""

import numpy as np

def compute_harm_rates(k_nc, N_nc, k_pcg, N_pcg):
    """Compute NoCert harm, PCG-MAS harm, raw gain, and Haldane-Anscombe corrected gain."""
    h_nc = k_nc / max(1, N_nc)
    h_pcg = k_pcg / max(1, N_pcg)
    
    raw_gain = h_nc / h_pcg if h_pcg > 0 else float('inf')
    haldane_anscombe_gain = ((k_nc + 0.5) * (N_pcg + 1)) / ((k_pcg + 0.5) * (N_nc + 1))
    
    return {
        "h_nc": h_nc,
        "h_pcg": h_pcg,
        "raw_gain": raw_gain,
        "haldane_anscombe_gain": haldane_anscombe_gain
    }

def compute_selectivity_verification(I_nc_all, H_nc_A, H_pcg_A):
    """Compute literal Selectivity (S) and Verification (V) decomposition."""
    S = I_nc_all - H_nc_A
    V = H_nc_A - H_pcg_A
    total_avoided = S + V
    residual = abs(total_avoided - (I_nc_all - H_pcg_A))
    return {"S": S, "V": V, "total_avoided": total_avoided, "residual": residual}
