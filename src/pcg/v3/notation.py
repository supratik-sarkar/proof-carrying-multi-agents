"""Single machine-readable notation registry.

Powers `table_33_notation.tex` AND the frontend glossary, so the manuscript's
notation can never drift from the symbols the code actually uses.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

#: (symbol, meaning, group)
SYMBOLS: List[Tuple[str, str, str]] = [
    (r"Z=(c,S,\Pi,\Gamma,p,\mathrm{meta})", "unified acceptance certificate", "Certificate"),
    (r"V_H", "evidence commitment / integrity conjunct", "Certificate"),
    (r"V_\Pi", "pinned-replay conjunct", "Certificate"),
    (r"V_\Gamma", "execution-contract conjunct", "Certificate"),
    (r"V_\vdash", "checker-relative entailment conjunct", "Certificate"),
    (r"\mathrm{Check}(Z;G_t)", r"$V_H\cdot V_\Pi\cdot V_\Gamma\cdot V_\vdash$", "Certificate"),
    (r"R_Z", "content-addressed certificate root (PCG-CAS-v1)", "Certificate"),
    (r"\mathsf{IntFail}", "commitment / logging audit channel", "Audit channels"),
    (r"\mathsf{ReplayFail}", "pinned-snapshot replay mismatch", "Audit channels"),
    (r"\mathsf{DriftFail}", "fresh-environment divergence", "Audit channels"),
    (r"\mathsf{CheckFail}", "unsound claim- or execution-side checking", "Audit channels"),
    (r"\mathsf{CovGap}", r"missing task/policy semantics in $(\mathcal R,\Gamma)$", "Audit channels"),
    (r"\varepsilon_{\mathrm{tax}}", "open-world taxonomy residual (outside every channel)", "Residuals"),
    (r"\varepsilon_{\mathrm{src}}", "source / world-truth residual (outside the certificate)", "Residuals"),
    (r"\pi_{\mathrm{unc}}", "uncovered deployment mass, charged once", "Audit sampling"),
    (r"B_{\mathrm{cov}}(\delta)", "covered-stratum envelope with inner sum clipped at 1", "Audit sampling"),
    (r"U_j(\delta)", "per-channel finite-sample envelope", "Audit sampling"),
    (r"\Lambda_\cup", r"union-overlap slack $\mathbb E[(N_F-1)_+]$", "Audit sampling"),
    (r"\lambda_{[k]}", "all-fail dependence ratio (likelihood ratio at the all-fail corner)", "Dependence"),
    (r"\rho_{[k]}", r"$\max\{1,\lambda_{[k]}^{1/(k-1)}\}$", "Dependence"),
    (r"\widehat\rho_{\mathrm{UCB}}", "monitored co-failure envelope over the subset lattice", "Dependence"),
    (r"U_{\mathrm{joint}}(k,\delta)", "direct binomial UCB for that exact $k$", "Dependence"),
    (r"q_{\mathrm{cm}}", "shared-cause mass; common-mode floor", "Dependence"),
    (r"S,\;V,\;\Delta", r"selectivity, verification, total gain; $\Delta=S+V$ exactly", "Decomposition"),
    (r"D_{\mathrm{alarm}}", r"$\max\{0,2a_{\mathrm{LCB}}-1\}$ shift alarm (not a bound)", "Shift"),
    (r"\mathsf{Resp}(e)", "replay-interventional attribution (not causal root cause)", "Diagnosis"),
    (r"\tau^\star", "ranking-recovery split (numerical minimiser)", "Diagnosis"),
    (r"L_{\mathrm{ctrl}},\;\epsilon_{\mathrm{cal}}", r"controller Lipschitz constant; calibration error", "Control"),
    (r"\tau_{\mathrm{instr}}", "instrumentation-overhead tolerance (0.05)", "Measurement"),
]


def groups() -> Dict[str, List[Tuple[str, str]]]:
    out: Dict[str, List[Tuple[str, str]]] = {}
    for sym, mean, grp in SYMBOLS:
        out.setdefault(grp, []).append((sym, mean))
    return out


def glossary() -> List[Dict[str, str]]:
    """Frontend glossary; same source as the manuscript table."""
    return [{"symbol": s, "meaning": m, "group": g} for s, m, g in SYMBOLS]
