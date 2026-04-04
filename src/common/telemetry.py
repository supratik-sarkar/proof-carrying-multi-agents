from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Optional


@dataclass
class StageStats:
    wall_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExampleTelemetry:
    run_id: str
    dataset: str
    instance_id: str
    backbone: str
    seed: int
    mode: str

    accepted: bool = False
    answer_correct: Optional[bool] = None
    decision: Optional[str] = None
    risk_raw: Optional[float] = None
    risk_cal: Optional[float] = None

    retrieval: StageStats = field(default_factory=StageStats)
    generation: StageStats = field(default_factory=StageStats)
    certificate: StageStats = field(default_factory=StageStats)
    replay: StageStats = field(default_factory=StageStats)
    verifier: StageStats = field(default_factory=StageStats)
    fallback: StageStats = field(default_factory=StageStats)

    total_wall_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0

    def finalize(self) -> None:
        stages = [
            self.retrieval,
            self.generation,
            self.certificate,
            self.replay,
            self.verifier,
            self.fallback,
        ]
        self.total_wall_ms = sum(s.wall_ms for s in stages)
        self.total_prompt_tokens = sum(s.prompt_tokens for s in stages)
        self.total_completion_tokens = sum(s.completion_tokens for s in stages)
        self.total_tokens = sum(s.total_tokens for s in stages)

    def to_dict(self) -> dict[str, Any]:
        self.finalize()
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "instance_id": self.instance_id,
            "backbone": self.backbone,
            "seed": self.seed,
            "mode": self.mode,
            "accepted": self.accepted,
            "answer_correct": self.answer_correct,
            "decision": self.decision,
            "risk_raw": self.risk_raw,
            "risk_cal": self.risk_cal,
            "retrieval": self.retrieval.to_dict(),
            "generation": self.generation.to_dict(),
            "certificate": self.certificate.to_dict(),
            "replay": self.replay.to_dict(),
            "verifier": self.verifier.to_dict(),
            "fallback": self.fallback.to_dict(),
            "total_wall_ms": self.total_wall_ms,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
        }
