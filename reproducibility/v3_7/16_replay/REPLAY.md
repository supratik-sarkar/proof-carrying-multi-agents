# Standalone replay

Run `verify_hashes.sh` from any location to verify record membership, the record manifest, the checksum list, the seal, every public projection, and the embedded manuscript stack. No path outside `reproducibility/v3_7` is required.

Run `rebuild_metrics.sh` to regenerate ten deterministic aggregate summaries from `14_artifacts/manuscript_stack/data` and compare every field with the v3.7 authority. Run `rebuild_artifacts.sh` to render 24 figure PDF/PNG pairs and 34 generated table files in a temporary directory and compare their hashes with the stored outputs.

Install the Python packages pinned in `requirements.txt` before replay. Figure replay also requires a TeX engine and a PDF rasterizer on `PATH`.

The preserved raw generations, trajectories, prompts, inputs, receipts, configuration, environment, and source-provenance records establish historical execution evidence through identity and hash reconciliation. Current per-observation verifier values, certificate values, decisions, and outcome labels were not retained. Those values are explicitly unavailable in sections 07–09 and are never inferred from aggregate data. Raw-to-aggregate scientific replay therefore remains unavailable; aggregate and artifact replay are deterministic and complete from the embedded v3.7 authority.
