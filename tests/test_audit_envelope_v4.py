import math
from types import SimpleNamespace

from pcg.eval.audit import (
    AUDIT_CHANNELS,
    estimate_audit_decomposition,
    estimate_audit_envelope_from_counts,
    hoeffding_upper_from_counts,
)


def test_hoeffding_upper_uses_channel_cardinality():
    failures = 5
    n = 100
    delta = 0.05
    num_channels = len(AUDIT_CHANNELS)

    item = hoeffding_upper_from_counts(
        failures=failures,
        n=n,
        delta=delta,
        num_channels=num_channels,
    )

    expected_beta = failures / n
    expected_radius = math.sqrt(math.log(num_channels / delta) / (2 * n))

    assert item.beta_hat == expected_beta
    assert item.radius == expected_radius
    assert item.U_delta == expected_beta + expected_radius


def test_audit_envelope_from_counts_has_total_row_semantics():
    counts = {
        "int": (1, 100),
        "rep": (2, 100),
        "chk": (3, 100),
        "cov": (4, 100),
    }

    envelope = estimate_audit_envelope_from_counts(counts, delta=0.05)

    assert tuple(envelope.channels) == AUDIT_CHANNELS
    assert set(envelope.per_channel) == set(AUDIT_CHANNELS)

    manual_sum = sum(item.U_delta for item in envelope.per_channel.values())
    assert envelope.sum_U_delta == manual_sum

    rows = envelope.rows()
    assert rows[-1]["channel"] == "TOTAL"
    assert rows[-1]["U_delta"] == envelope.sum_U_delta


def test_audit_decomposition_backward_compatible_without_unsafe_execution():
    results = [
        SimpleNamespace(
            passed=True,
            integrity_ok=True,
            replay_ok=True,
            entailment_ok=True,
            execution_ok=True,
        ),
        SimpleNamespace(
            passed=False,
            integrity_ok=False,
            replay_ok=True,
            entailment_ok=True,
            execution_ok=True,
        ),
        SimpleNamespace(
            passed=False,
            integrity_ok=True,
            replay_ok=False,
            entailment_ok=True,
            execution_ok=True,
        ),
        SimpleNamespace(
            passed=False,
            integrity_ok=True,
            replay_ok=True,
            entailment_ok=False,
            execution_ok=True,
        ),
    ]
    gt_correct = [False, True, True, True]

    decomp = estimate_audit_decomposition(
        results,
        gt_correct,
        envelope_delta=0.05,
    )

    assert decomp.n == 4
    assert decomp.lhs_accept_and_bad == decomp.lhs_accept_and_wrong
    assert decomp.raw["lhs_bad"] == [True, False, False, False]
    assert decomp.raw["int_fail"] == [False, True, False, False]
    assert decomp.raw["replay_fail"] == [False, False, True, False]
    assert decomp.raw["check_fail"] == [False, False, False, True]
    assert decomp.raw["cov_gap"] == [True, False, False, False]
    assert decomp.envelope is not None


def test_audit_decomposition_counts_accepted_unsafe_execution_as_bad():
    results = [
        SimpleNamespace(
            passed=True,
            integrity_ok=True,
            replay_ok=True,
            entailment_ok=True,
            execution_ok=True,
        ),
    ]

    decomp = estimate_audit_decomposition(
        results,
        ground_truth_correct=[True],
        unsafe_execution=[True],
        envelope_delta=0.05,
    )

    assert decomp.raw["lhs_bad"] == [True]
    assert decomp.raw["cov_gap"] == [True]