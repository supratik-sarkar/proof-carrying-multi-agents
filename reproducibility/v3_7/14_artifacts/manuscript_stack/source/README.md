# Manuscript artifact source

All repeatable visual choices are stored here, including Python plotting, vector diagram construction and the TeX theory-dependency diagram. See `../DATA_REFRESH.md` for the complete rebuild and data-refresh contract.

- `figure_contract.py`: stable artifact names and input dependencies. Emitter identifiers are internal lookup keys; manuscript figure numbers come from the LaTeX labels.
- `figure_01_layout.py`, `figure_labels.py`: executive atlas, seven-cell distribution summaries and collision-aware aggregate labels.
- `protocol_figures.py`: runtime workflow and explicit certificate/trust-boundary notation.
- `main_result_layouts.py`: release-control funnel and adversarial/shift atlas.
- `figure_03_04_layouts.py`: appendix theory geometry and main mechanism atlas, including the guarded decomposition branch.
- `render_result_figures.py`, `render_static_schematics.py`: common result rendering and source-authored diagrams.
- `figure_B1_theory_dependency.tex`, `build_theory_dependency.py`: shared diagram source with references resolved from the compiled manuscript.
- `prepare_main_tables.py`, `generate_tables.py`, `table_contract.json`: derived tables and preserved captions/labels. The five detailed notation tables are authored TeX references.
- `build_figure_architecture.py`, `compile_manuscript.py`: deterministic orchestration, normal pdfLaTeX/BibTeX compilation, and figure manifests.
- `check_figure_01.py`, `check_figure_03_04.py`, `check_appendix_figures.py`, `check_data_refresh.py`: numerical, protocol-branch and same-input reproduction checks.
- `validate_manuscript_artifacts.py`, `validate_package.py`, `check_bibliography.py`: artifact completeness, sources, compiled references, embedded fonts, page budget and bibliography closure.

The source preserves both sign conventions explicitly: positive executive reduction and negative-favors-PCG primary risk difference. It does not synthesize undeclared operating points, recover unavailable raw records, or silently fill missing values with zero. The source accepts values-only updates under the existing scientific and category contracts.
