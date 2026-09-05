# MAC_M4_RUNBOOK.md — MacBook Pro M4 Pro

## Setup

```bash
cd ~/Desktop/pcg-mas-2026
python3.12 -m venv .venv-pcg-mas
source .venv-pcg-mas/bin/activate
python -V                                  # Python 3.12.13
python -m pip install -U pip setuptools wheel
python -m pip install -r env/constraints-offline.txt
python -m pip install -e .
bash scripts/v3/verify_offline.sh
```

Model extras are installed only when you reach a workstream that needs them. The eleven offline workstreams need none of it.

## Device routing

```
CUDA → MPS → CPU        # on this machine: MPS → CPU
```

`providers/base.resolve_device()` checks `torch.backends.mps.is_available()` at **runtime**. No CUDA-only API is touched at import, so every module imports on a machine with no GPU. The resolved device is written into each run's `environment.json`. An operator unsupported on MPS falls back to CPU explicitly and is logged — never silently.

## What belongs here

All eleven offline workstreams (A02, A03, A04, A06, A07, A11, A12, A13, A14, A17, A18) plus the small-backend cells of A01/A05/A15. This is the right machine for Gate 1 and Gate 2 — the decisions that determine whether the story survives.

## What does not

70B and 671B cells. Route them to Colab and record the provider route in the backend fingerprint.

## Timing caveat

A10 timing measured here is **not comparable** to Colab timing and must never be pooled with it. Report Mac and Colab rows separately with the device fingerprint attached. Do not convert local wall time into a dollar figure and present it as measured cost; an optional cloud-price conversion is `MODELLED`, never `DIRECT`.

## Demo locally

```bash
python app/shared/generate_contract.py
PYTHONPATH=src python app/backend/main.py     # http://127.0.0.1:8000
```

Runs with or without FastAPI installed.

## Housekeeping

Never archive `.venv-pcg-mas`. Do not restore `__pycache__` from a 3.10 interpreter. Rebuild an upload archive with the exclusions in `V3_0_HANDOFF.md`.
