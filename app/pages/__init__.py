"""PCG-MAS demo pages.

This file makes `pages` an importable package so cross-page helpers in
`_live_run_helpers.py` can be imported via `from pages._live_run_helpers
import ...`. Streamlit's page-discovery mechanism still treats every
non-underscore-prefixed `*.py` file in this directory as a navigable page.
"""
