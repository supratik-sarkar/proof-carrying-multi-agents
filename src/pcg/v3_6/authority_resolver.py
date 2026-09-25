"""PCG-MAS v3.6 Authority Resolver.

Provides canonical path resolution and constant authorities for the cleaned
reproducible repository structure.
"""

import json
from pathlib import Path
from typing import Optional, Union

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Core Output Directories
OUTPUTS_V3_6 = (
    REPO_ROOT / "Bucket_2" / "outputs" / "v3_6"
    if (REPO_ROOT / "Bucket_2" / "outputs" / "v3_6").exists()
    else REPO_ROOT / "outputs" / "v3_6"
)
RAW_DIR = OUTPUTS_V3_6 / "raw"
RECEIPTS_DIR = OUTPUTS_V3_6 / "receipts"
MANIFESTS_DIR = OUTPUTS_V3_6 / "manifests"

# Scientific Inputs
CANONICAL_GENERATIONS_DIR = RAW_DIR / "generations"
CANONICAL_2240_REGISTRY = CANONICAL_GENERATIONS_DIR / "REGISTRY.jsonl"
CANONICAL_2240_MANIFEST = CANONICAL_GENERATIONS_DIR / "CANONICAL_2240_MANIFEST.json"

CERTIFICATION_INPUTS_DIR = RAW_DIR / "certification_inputs"
FUSION_FIT_RECORDS_PATH = CERTIFICATION_INPUTS_DIR / "FUSION_FIT_RECORDS_V2.jsonl"
FACTUAL_VERIFIER_INPUTS_PATH = CERTIFICATION_INPUTS_DIR / "FACTUAL_VERIFIER_INPUTS_1400.jsonl"
FACTUAL_CANDIDATE_ELIGIBILITY_PATH = CERTIFICATION_INPUTS_DIR / "FACTUAL_CANDIDATE_ELIGIBILITY_1400.jsonl"

EXTERNAL_VERIFIERS_DIR = RAW_DIR / "external_verifiers"
EXTERNAL_VERIFIER_SLOTS_PATH = EXTERNAL_VERIFIERS_DIR / "EXTERNAL_VERIFIER_SLOTS_4200.jsonl"
EXTERNAL_VERIFIER_SCORES_PATH = EXTERNAL_VERIFIERS_DIR / "EXTERNAL_VERIFIER_SCORES_4200.jsonl"
COMMON_SUPPORT_SCORES_PATH = EXTERNAL_VERIFIERS_DIR / "COMMON_SUPPORT_EXTERNAL_VERIFIER_SCORES_4197.jsonl"
MINICHECK_SCORES_PATH = EXTERNAL_VERIFIERS_DIR / "MINICHECK_SCORES_1400.jsonl"
ALIGNSCORE_SCORES_PATH = EXTERNAL_VERIFIERS_DIR / "ALIGNSCORE_SCORES_1400.jsonl"
QAFACTEVAL_SCORES_PATH = EXTERNAL_VERIFIERS_DIR / "QAFACTEVAL_SCORES_1400.jsonl"

COMPARATORS_DIR = RAW_DIR / "comparators"
CORE_COMPARATOR_FIT_PATH = COMPARATORS_DIR / "CORE_COMPARATOR_FIT_RESULT.json"
COMPARATOR_STATUS_PATH = COMPARATORS_DIR / "COMPARATOR_STATUS.json"

AGENTDOJO_RUNS_DIR = RAW_DIR / "agentdojo"

# Model Checkpoints
DEBERTA_CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints" / "pcg_nli_deberta_v3_large"

# Configurations & Authorities
AUTHORITY_CONFIGS_DIR = REPO_ROOT / "configs" / "v3_6" / "authority"
VERIFIER_APPLICABILITY_PATH = AUTHORITY_CONFIGS_DIR / "VERIFIER_APPLICABILITY.json"
COMPARATOR_APPLICABILITY_PATH = AUTHORITY_CONFIGS_DIR / "COMPARATOR_APPLICABILITY.json"
CERTIFICATION_PROFILE_MAP_PATH = AUTHORITY_CONFIGS_DIR / "CERTIFICATION_PROFILE_MAP.json"
SOURCE_PINS_PATH = AUTHORITY_CONFIGS_DIR / "SOURCE_PINS.json"
GENERATION_POLICY_REGISTRY_PATH = AUTHORITY_CONFIGS_DIR / "GENERATION_POLICY_REGISTRY.json"
DATASET_PANEL_PATH = AUTHORITY_CONFIGS_DIR / "DATASET_PANEL.json"

# Pipeline Scripts
PIPELINE_SCRIPTS_DIR = REPO_ROOT / "scripts" / "v3_6" / "pipeline"
CERTIFY_V2_SCRIPT = PIPELINE_SCRIPTS_DIR / "certify_v2.py"
FIT_V2_SCRIPT = PIPELINE_SCRIPTS_DIR / "fit_v2.py"
COMPARATORS_V2_SCRIPT = PIPELINE_SCRIPTS_DIR / "comparators_v2.py"

# Experiment Specifications
FIGURE_EXECUTION_SPEC_PATH = REPO_ROOT / "docs" / "experiment_specifications" / "PCG_MAS_V3_6_FIGURE_EXECUTION_SPECIFICATION.md"

# Load migration map if available
_MIGRATION_MAP: Optional[dict] = None

def _get_migration_map() -> dict:
    global _MIGRATION_MAP
    if _MIGRATION_MAP is None:
        map_path = REPO_ROOT / "Bucket_2" / "PATH_MIGRATION_MAP.json"
        if not map_path.is_file():
            map_path = REPO_ROOT / "PATH_MIGRATION_MAP.json"
        if map_path.is_file():
            try:
                with open(map_path, "r", encoding="utf-8") as f:
                    _MIGRATION_MAP = json.load(f).get("entries", {})
            except Exception:
                _MIGRATION_MAP = {}
        else:
            _MIGRATION_MAP = {}
    return _MIGRATION_MAP


def resolve_path(path: Union[str, Path]) -> Path:
    """Resolve a path to its canonical repository location.
    
    If given a legacy path that has been migrated, returns the new canonical Path.
    Otherwise returns the path resolved relative to REPO_ROOT if relative, or as-is if absolute.
    """
    p = Path(path)
    if p.is_absolute():
        try:
            rel = str(p.relative_to(REPO_ROOT))
        except ValueError:
            return p
    else:
        rel = str(p)

    m = _get_migration_map()
    if rel in m:
        canonical_rel = m[rel].get("canonical_path", rel)
        cand = REPO_ROOT / "Bucket_2" / canonical_rel
        if cand.exists():
            return cand
        return REPO_ROOT / canonical_rel

    candidate = REPO_ROOT / rel
    if candidate.exists():
        return candidate
    candidate_b2 = REPO_ROOT / "Bucket_2" / rel
    if candidate_b2.exists():
        return candidate_b2

    return p
