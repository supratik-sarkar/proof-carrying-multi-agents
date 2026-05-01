"""Audit decomposition and finite-sample audit envelope for PCG-MAS v4.

This module implements the empirical objects behind Theorem 5.1:

    Bad_t(c) subset IntFail_t union ReplayFail_t union CheckFail_t union CovGap_t

and the finite-sample envelope

    Pr(Bad_t(c)) <= sum_{j in J} U_j(delta),

where J = {int, rep, chk, cov} and

    U_j(delta) = beta_hat_j + sqrt(log(|J| / delta) / (2 n_j)).

The implementation intentionally keeps |J| symbolic as len(AUDIT_CHANNELS), not as a hard-coded 4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from pcg.eval.stats import wilson_interval


AUDIT_CHANNELS: tuple[str, ...] = ("int", "rep", "chk", "cov")


@dataclass(frozen=True)
class AuditChannelEnvelope:
    """Finite-sample upper envelope for one audit channel."""

    channel: str
    n: int
    failures: int
    beta_hat: float
    radius: float
    U_delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "n": self.n,
            "failures": self.failures,
            "beta_hat": self.beta_hat,
            "radius": self.radius,
            "U_delta": self.U_delta,
        }


@dataclass(frozen=True)
class AuditEnvelope:
    """Simultaneous finite-sample audit envelope over all channels."""

    delta: float
    channels: tuple[str, ...]
    per_channel: dict[str, AuditChannelEnvelope]
    sum_U_delta: float

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "delta": self.delta,
            "channels": list(self.channels),
            "sum_U_delta": self.sum_U_delta,
        }
        for channel, item in self.per_channel.items():
            out[channel] = item.to_dict()
        return out

    def rows(self) -> list[dict[str, Any]]:
        rows = [self.per_channel[channel].to_dict() for channel in self.channels]
        rows.append(
            {
                "channel": "TOTAL",
                "n": "",
                "failures": "",
                "beta_hat": "",
                "radius": "",
                "U_delta": self.sum_U_delta,
            }
        )
        return rows


@dataclass
class AuditDecomposition:
    """Empirical realization of the four audit-channel decomposition.

    Each p_* field is the Wilson-score point estimate for the corresponding channel.
    lhs_accept_and_bad is the empirical bad-accept event:
        accepted and (false claim or unsafe execution).

    The legacy alias lhs_accept_and_wrong is kept for compatibility with older scripts.
    """

    n: int

    p_int_fail: float
    ci_int_fail: tuple[float, float]

    p_replay_fail: float
    ci_replay_fail: tuple[float, float]

    p_check_fail: float
    ci_check_fail: tuple[float, float]

    p_cov_gap: float
    ci_cov_gap: tuple[float, float]

    lhs_accept_and_bad: float
    ci_lhs: tuple[float, float]

    rhs_union: float
    envelope: AuditEnvelope | None = None
    raw: dict[str, list[bool]] = field(default_factory=dict)

    @property
    def lhs_accept_and_wrong(self) -> float:
        """Backward-compatible name used by older R1 code."""
        return self.lhs_accept_and_bad

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n": self.n,
            "lhs_accept_and_bad": self.lhs_accept_and_bad,
            "lhs_accept_and_wrong": self.lhs_accept_and_bad,
            "ci_lhs": list(self.ci_lhs),
            "rhs_union": self.rhs_union,
            "p_int_fail": self.p_int_fail,
            "ci_int_fail": list(self.ci_int_fail),
            "p_replay_fail": self.p_replay_fail,
            "ci_replay_fail": list(self.ci_replay_fail),
            "p_check_fail": self.p_check_fail,
            "ci_check_fail": list(self.ci_check_fail),
            "p_cov_gap": self.p_cov_gap,
            "ci_cov_gap": list(self.ci_cov_gap),
        }
        if self.envelope is not None:
            out["audit_envelope"] = self.envelope.to_dict()
        return out


def hoeffding_upper_from_counts(
    *,
    failures: int,
    n: int,
    delta: float,
    num_channels: int,
) -> AuditChannelEnvelope:
    """Compute U_j(delta) from counts for one audit channel.

    Args:
        failures: Number of failed probes for the channel.
        n: Number of audit probes for the channel.
        delta: Total failure probability for the simultaneous envelope.
        num_channels: Number of simultaneous audit channels, i.e. |J|.

    Returns:
        AuditChannelEnvelope with beta_hat, Hoeffding radius, and U_delta.
    """
    if n <= 0:
        raise ValueError("n must be positive for a finite-sample audit envelope")
    if failures < 0 or failures > n:
        raise ValueError(f"failures must satisfy 0 <= failures <= n, got {failures}/{n}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must lie in (0,1), got {delta}")
    if num_channels <= 0:
        raise ValueError("num_channels must be positive")

    beta_hat = failures / n
    radius = math.sqrt(math.log(num_channels / delta) / (2.0 * n))
    return AuditChannelEnvelope(
        channel="",
        n=n,
        failures=failures,
        beta_hat=beta_hat,
        radius=radius,
        U_delta=beta_hat + radius,
    )


def estimate_audit_envelope_from_counts(
    counts: Mapping[str, tuple[int, int]],
    *,
    delta: float = 0.05,
    channels: Sequence[str] = AUDIT_CHANNELS,
) -> AuditEnvelope:
    """Estimate the simultaneous audit envelope from channel counts.

    Args:
        counts: Mapping channel -> (failures, n).
        delta: Total failure probability for the simultaneous envelope.
        channels: Audit channel set J.

    Returns:
        AuditEnvelope containing U_j(delta) for each channel and sum_j U_j(delta).
    """
    channel_tuple = tuple(channels)
    missing = [channel for channel in channel_tuple if channel not in counts]
    if missing:
        raise ValueError(f"missing audit-channel counts for: {missing}")

    per_channel: dict[str, AuditChannelEnvelope] = {}
    for channel in channel_tuple:
        failures, n = counts[channel]
        item = hoeffding_upper_from_counts(
            failures=int(failures),
            n=int(n),
            delta=delta,
            num_channels=len(channel_tuple),
        )
        per_channel[channel] = AuditChannelEnvelope(
            channel=channel,
            n=item.n,
            failures=item.failures,
            beta_hat=item.beta_hat,
            radius=item.radius,
            U_delta=item.U_delta,
        )

    return AuditEnvelope(
        delta=delta,
        channels=channel_tuple,
        per_channel=per_channel,
        sum_U_delta=sum(item.U_delta for item in per_channel.values()),
    )


def estimate_audit_envelope_from_flags(
    flags: Mapping[str, Sequence[bool]],
    *,
    delta: float = 0.05,
    channels: Sequence[str] = AUDIT_CHANNELS,
) -> AuditEnvelope:
    """Estimate the finite-sample audit envelope from Boolean channel-failure flags."""
    counts: dict[str, tuple[int, int]] = {}
    for channel in channels:
        arr = np.asarray(flags[channel], dtype=bool)
        counts[channel] = (int(arr.sum()), int(arr.size))
    return estimate_audit_envelope_from_counts(counts, delta=delta, channels=channels)


def estimate_audit_decomposition(
    check_results: Sequence[Any],
    ground_truth_correct: Sequence[bool],
    *,
    unsafe_execution: Sequence[bool] | None = None,
    alpha: float = 0.05,
    envelope_delta: float | None = None,
) -> AuditDecomposition:
    """Estimate R1 audit decomposition and optionally Theorem 5.1 envelope.

    Args:
        check_results: One CheckResult-like object per claim. Required fields:
            passed, integrity_ok, replay_ok, entailment_ok, execution_ok.
        ground_truth_correct: True iff the emitted claim is correct under task labels.
        unsafe_execution: Optional flag for accepted execution that violates intended semantics
            but was not caught by the declared execution contract. If omitted, all False.
        alpha: Wilson interval level, where confidence is 1-alpha.
        envelope_delta: If provided, also compute U_j(envelope_delta).

    Returns:
        AuditDecomposition with raw channel flags and optional AuditEnvelope.
    """
    n = len(check_results)
    if n != len(ground_truth_correct):
        raise ValueError("check_results and ground_truth_correct must align")

    if unsafe_execution is None:
        unsafe = np.zeros(n, dtype=bool)
    else:
        if n != len(unsafe_execution):
            raise ValueError("unsafe_execution and check_results must align")
        unsafe = np.asarray(unsafe_execution, dtype=bool)

    gt_correct = np.asarray(ground_truth_correct, dtype=bool)

    int_fail = np.zeros(n, dtype=bool)
    replay_fail = np.zeros(n, dtype=bool)
    check_fail = np.zeros(n, dtype=bool)
    cov_gap = np.zeros(n, dtype=bool)
    lhs_bad = np.zeros(n, dtype=bool)

    for i, (result, gt_ok) in enumerate(zip(check_results, gt_correct)):
        integrity_ok = bool(getattr(result, "integrity_ok"))
        replay_ok = bool(getattr(result, "replay_ok"))
        entailment_ok = bool(getattr(result, "entailment_ok"))
        execution_ok = bool(getattr(result, "execution_ok"))
        passed = bool(getattr(result, "passed"))

        int_fail[i] = not integrity_ok
        replay_fail[i] = integrity_ok and not replay_ok

        # CheckFail is a checker-side failure after commitment and replay are intact.
        check_fail[i] = integrity_ok and replay_ok and not (entailment_ok and execution_ok)

        # CovGap is the residual semantic-coverage gap: the certificate passed,
        # but the run is actually bad under external task or policy semantics.
        bad_semantics = (not gt_ok) or bool(unsafe[i])
        cov_gap[i] = passed and bad_semantics
        lhs_bad[i] = passed and bad_semantics

    def _wilson(flags: np.ndarray) -> tuple[float, tuple[float, float]]:
        k = int(flags.sum())
        p_hat, lo, hi = wilson_interval(k, n, alpha=alpha)
        return p_hat, (lo, hi)

    p_int, ci_int = _wilson(int_fail)
    p_rep, ci_rep = _wilson(replay_fail)
    p_chk, ci_chk = _wilson(check_fail)
    p_cov, ci_cov = _wilson(cov_gap)
    p_lhs, ci_lhs = _wilson(lhs_bad)

    raw = {
        "int_fail": int_fail.tolist(),
        "replay_fail": replay_fail.tolist(),
        "check_fail": check_fail.tolist(),
        "cov_gap": cov_gap.tolist(),
        "lhs_bad": lhs_bad.tolist(),
        "lhs": lhs_bad.tolist(),  # legacy alias
    }

    envelope = None
    if envelope_delta is not None:
        envelope = estimate_audit_envelope_from_flags(
            {
                "int": int_fail,
                "rep": replay_fail,
                "chk": check_fail,
                "cov": cov_gap,
            },
            delta=envelope_delta,
            channels=AUDIT_CHANNELS,
        )

    return AuditDecomposition(
        n=n,
        p_int_fail=p_int,
        ci_int_fail=ci_int,
        p_replay_fail=p_rep,
        ci_replay_fail=ci_rep,
        p_check_fail=p_chk,
        ci_check_fail=ci_chk,
        p_cov_gap=p_cov,
        ci_cov_gap=ci_cov,
        lhs_accept_and_bad=p_lhs,
        ci_lhs=ci_lhs,
        rhs_union=p_int + p_rep + p_chk + p_cov,
        envelope=envelope,
        raw=raw,
    )