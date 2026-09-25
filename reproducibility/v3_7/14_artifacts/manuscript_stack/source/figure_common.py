"""Compatibility dispatch to the canonical data-contract renderer."""
from figure_contract import NAMES, STATIC
def generate(name):
    n=NAMES.index(name)+1
    if n in STATIC:
        from render_static_schematics import FUNCTIONS
        FUNCTIONS[n]()
    else:
        from render_result_figures import generate as render
        render(n)
