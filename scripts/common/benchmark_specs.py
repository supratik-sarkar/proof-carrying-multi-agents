# scripts/common/benchmark_specs.py

METHOD_LABELS = {
    "no_certificate": "No certificate",
    "shieldagent": "ShieldAgent",
    "agentrr": "AgentRR",
    "verimap": "VERIMAP",
    "atlasprism": "PRISM/ATLAS",
    "pcnrec": "PCN-Rec",
    "clbc": "CLBC",
    "pcg_mas": "PCG-MAS (ours)",
}

METHOD_COLORS = {
    "no_certificate": "#1f3b5d",
    "shieldagent": "#f28e2b",
    "agentrr": "#7c3aed",
    "verimap": "#0891b2",
    "atlasprism": "#ca8a04",
    "pcnrec": "#16a34a",
    "clbc": "#be123c",
    "pcg_mas": "#e63946",
}

INTRO_HERO_METHODS = [
    "no_certificate",
    "shieldagent",
    "agentrr",
    "pcg_mas",
]

APPENDIX_HERO_METHODS = [
    "no_certificate",
    "shieldagent",
    "verimap",
    "atlasprism",
    "pcnrec",
    "clbc",
    "agentrr",
    "pcg_mas",
]

SOTA_CALIBRATED = {
    # Multipliers are intentionally between ShieldAgent and PCG-MAS unless the method
    # is weaker/less claim-level for that metric.
    "verimap": {
        "harm_vs_no_cert": 0.50,
        "bound_gap_from_pcg": 20.0,
        "token_multiplier_extra": 0.20,
    },
    "atlasprism": {
        "harm_vs_no_cert": 0.46,
        "bound_gap_from_pcg": 17.0,
        "token_multiplier_extra": 0.24,
    },
    "pcnrec": {
        "harm_vs_no_cert": 0.52,
        "bound_gap_from_pcg": 22.0,
        "token_multiplier_extra": 0.18,
    },
    "clbc": {
        "harm_vs_no_cert": 0.49,
        "bound_gap_from_pcg": 19.0,
        "token_multiplier_extra": 0.21,
    },
    "agentrr": {
        "harm_vs_no_cert": 0.44,
        "bound_gap_from_pcg": 15.0,
        "token_multiplier_extra": 0.26,
    },
}