# PCG-MAS v3.7 public reproducibility record

This directory is the canonical public v3.7 record. Its 00–16 structure contains the v3.7 protocol, embedded numerical authority, rendering source, reported artifacts, and replay tools, together with hash-reconciled historical execution evidence.

The sealed v3.6 record is used only as an evidence donor. Its raw generations, trajectories, prompts, inputs, execution receipts, configuration, environment, and source-provenance records are admitted only after identity and SHA-256 reconciliation. The donor manifest, seal, and checksum-list hashes are recorded in `15_change_control/v3_6_donor_binding.json`; each admitted artifact is recorded in `15_change_control/donor_artifact_binding.jsonl` with its historical-source hash, public-projection hash, and redaction reason. No v3.6 scientific result value from verification, certificates, outcomes, metrics, or artifacts is authoritative here.

The sole scientific and results authority is `14_artifacts/manuscript_stack/data`. The corresponding source, figure files, and table files are stored beside it, allowing aggregate and artifact replay without the private project tree. Sections 13 and 14 bind every numerical input and rendered artifact to these embedded bytes.

Current per-observation verifier values, certificate values, acceptance decisions, and outcome labels were not retained in the surviving v3.7 material. Sections 07–09 enumerate the affected identities with `NOT_RETAINED`; they contain no reconstructed values. This explicit gap prevents a raw-to-aggregate scientific replay claim while preserving complete evidence about what is and is not available.

Run `sh 16_replay/verify_hashes.sh`, `sh 16_replay/rebuild_metrics.sh`, and `sh 16_replay/rebuild_artifacts.sh` from this directory or invoke the same paths from a repository root. See `16_replay/REPLAY.md` for scope.
