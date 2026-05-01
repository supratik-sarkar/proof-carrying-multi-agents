from __future__ import annotations

METHODS = ["no_certificate", "shieldagent", "pcg_mas"]

METHOD_LABELS = {
    "no_certificate": "No certificate",
    "shieldagent": "SHIELDAGENT",
    "pcg_mas": "PCG-MAS (ours)",
}

METHOD_COLORS = {
    "no_certificate": "#1f3b5d",   # dark navy
    "shieldagent": "#f28e2b",      # orange
    "pcg_mas": "#e63946",          # red
}

CHANNEL_LABELS = {
    "integrity": "Integrity",
    "replay": "Replay",
    "check": "Checker",
    "coverage": "Coverage",
}

CHANNEL_COLORS = {
    "integrity": "#264653",
    "replay": "#2a9d8f",
    "check": "#e9c46a",
    "coverage": "#f4a261",
}

R_PLOT_CELLS = [
    ("qwen2.5-7B", "HotpotQA"),
    ("Llama-3.1-8B", "PubMedQA"),
    ("deepseek-v3", "WebLINX"),
]

MAIN6_CELLS = [
    ("phi-3.5-mini", "FEVER"),
    ("qwen2.5-7B", "HotpotQA"),
    ("Llama-3.1-8B", "PubMedQA"),
    ("Gemma-2-9b-it", "TAT-QA"),
    ("Llama-3.3-70B", "ToolBench"),
    ("deepseek-v3", "WebLINX"),
]

ALL_DATASETS = [
    "HotpotQA",
    "2WikiMultihopQA",
    "TAT-QA",
    "ToolBench",
    "FEVER",
    "PubMedQA",
    "WebLINX",
    "Synthetic-Adversarial",
]

ALL_MODELS = [
    "phi-3.5-mini",
    "qwen2.5-7B",
    "deepseek-llm-7b-chat",
    "Llama-3.1-8B",
    "Gemma-2-9b-it",
    "Llama-3.3-70B",
    "deepseek-v3",
]

FIG_FONT = {
    "title": 27,
    "subtitle": 20,
    "axis": 21,
    "tick": 18,
    "legend": 17,
    "annot": 18,
    "small": 15,
}

FIG_STYLE = {
    "ink": "#020617",
    "ink_light": "#334155",
    "grid": "#64748b",
    "edge": "#111827",
}