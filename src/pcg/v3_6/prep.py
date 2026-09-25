"""Single prompt/evidence preparation path.

The prompt is NOT rebuilt here. It is captured from the prover's own code by
running the real prover with a backend that records the prompt and aborts
before generating. One implementation, used by both stages, so the dispatcher
and the certification replay cannot drift.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional

from pcg.agents.prover import ProverConfig, build_default_prover
from pcg.orchestrator.langgraph_flow import PCGState

from .hashing36 import sha256_json, sha256_text

PREP_SCHEMA = "PCG_MAS_V3_6_GENERATION_PREP_V1"


class PromptCaptured(Exception):
    def __init__(self, prompt: str, kwargs: Dict[str, Any]):
        super().__init__("PROMPT_CAPTURED")
        self.prompt = prompt
        self.kwargs = kwargs


class CapturePromptBackend:
    """Records the prompt the prover builds, then aborts. Never generates."""

    name = "capture-prompt-v3.6"

    def __init__(self) -> None:
        self.prompt: Optional[str] = None
        self.kwargs: Dict[str, Any] = {}

    def generate(self, prompt: str, **kwargs: Any):
        self.prompt = prompt
        self.kwargs = dict(kwargs)
        raise PromptCaptured(prompt, self.kwargs)


def evidence_identity(example) -> Dict[str, Any]:
    """Content address of the example's evidence pool, order preserved."""
    items = []
    for ev in example.evidence:
        items.append({
            "id": getattr(ev, "id", None),
            "title": getattr(ev, "title", None),
            "text": getattr(ev, "text", None),
            "publisher": getattr(ev, "publisher", None),
        })
    return {"evidence_count": len(items), "evidence_sha256": sha256_json(items)}


def prepare(example, config: Optional[ProverConfig] = None) -> Dict[str, Any]:
    """Return the exact prompt the prover will build for this example.

    The returned prompt_sha256 is the contract between generation and
    certification. It binds retrieval as well, because the retrieved context is
    interpolated into the prompt text.
    """
    cfg = config or ProverConfig()
    cap = CapturePromptBackend()
    prover = build_default_prover(backend=cap, config=cfg)
    state = PCGState(example=example)
    try:
        prover(state)
    except PromptCaptured as pc:
        ev = evidence_identity(example)
        return {
            "schema": PREP_SCHEMA,
            "prompt": pc.prompt,
            "prompt_sha256": sha256_text(pc.prompt),
            "generation_kwargs": pc.kwargs,
            "prover_config": {
                "top_k": cfg.top_k, "max_answer_tokens": cfg.max_answer_tokens,
                "temperature": cfg.temperature, "seed": cfg.seed,
                "retriever": cfg.retriever, "prompt_variant": cfg.prompt_variant,
                "use_concat_replay": cfg.use_concat_replay,
            },
            "task_type": getattr(example, "task_type", None),
            "example_id": getattr(example, "id", None),
            **ev,
        }
    raise RuntimeError("PREP_DID_NOT_REACH_GENERATION: prover returned without calling generate")
