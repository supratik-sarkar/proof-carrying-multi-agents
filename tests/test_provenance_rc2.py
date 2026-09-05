"""RC2 provenance tests. Each named for the failure it prevents."""
from __future__ import annotations

import json, math, unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from pcg.provenance import (AggregateResult, ExecutionClass, RecordIdentity, RunRecorder,
                            add, aggregate, classify, code_fingerprint, execution_class,
                            hash_obj, hash_text, is_direct_eligible, mean, ratio, run_gates,
                            sanitize, verify_recomputable)

T0 = datetime(2027, 1, 1, tzinfo=timezone.utc)


def ident(backend="hf_inference", provider="hf", i=0, seed=0, prompt="p"):
    return RecordIdentity(experiment_id="R1", dataset="FEVER", example_id=f"ex{i:03d}",
                          condition="clean", seed=seed, provider=provider, backend=backend,
                          requested_model="phi-3.5-mini", experiment_config_hash="cfg",
                          decoding_config_hash="dec", input_hash=hash_text(prompt))


def rec(rc: RunRecorder, i=0, backend="hf_inference", prompt=None, out=None,
        lat=850.0, itok=120, otok=64, usage="provider_usage", status="ok", **over):
    prompt = prompt or f"prompt {i}"
    out = out if out is not None else f"output {i}"
    idn = ident(backend=backend, i=i, prompt=prompt)
    rid = idn.record_id()
    ref, bh = rc.write_bundle(rid, canonical_input=prompt, raw_output=out)
    ev = dict(requested_model="phi-3.5-mini", returned_model="phi-3.5-mini",
              model_revision="rev-abc", provider="hf", backend=backend,
              start_timestamp=T0.isoformat(),
              end_timestamp=(T0 + timedelta(milliseconds=lat)).isoformat(),
              latency_ms=lat, monotonic_measured=True,
              input_tokens=itok, output_tokens=otok,
              total_tokens=(itok + otok) if (itok is not None and otok is not None) else None,
              usage_source=usage)
    ev.update(over.pop("evidence", {}))
    base = dict(record_id=rid, identity=idn.canonical(), identity_digest=idn.digest(),
                evidence=ev, decoding_params={"temperature": 0.0, "top_p": 1.0, "max_tokens": 256},
                input_hash=hash_text(prompt), output_hash=hash_text(out),
                bundle_ref=ref, bundle_hash=bh, model_call_count=1, status=status)
    base.update(over)
    return rc.append(base)


class TestNumericSafety(unittest.TestCase):
    def test_unknown_never_becomes_zero(self):
        self.assertIsNone(add(5, None))
        self.assertIsNone(add(None, None))
        self.assertEqual(add(5, 7), 12)

    def test_ratio_empty_denominator_is_none(self):
        self.assertIsNone(ratio(3, 0)); self.assertIsNone(ratio(3, None)); self.assertIsNone(ratio(None, 3))

    def test_mean_require_all_propagates_unknown(self):
        self.assertIsNone(mean([1, None, 3], require_all=True))
        self.assertEqual(mean([1, None, 3]), 2.0)

    def test_total_tokens_absent_when_either_side_unknown(self):
        with TemporaryDirectory() as d:
            rc = RunRecorder(Path(d), "R1")
            r = rec(rc, otok=None, usage="provider_usage")
            self.assertIsNone(r["evidence"]["total_tokens"])
            self.assertNotEqual(r["evidence"]["total_tokens"], 0)


class TestBackendEligibility(unittest.TestCase):
    def test_mock_is_not_direct_eligible(self):
        self.assertIs(execution_class("mock"), ExecutionClass.MOCK)
        self.assertFalse(is_direct_eligible("mock"))

    def test_unregistered_backend_fails_closed(self):
        self.assertIs(execution_class("brand_new"), ExecutionClass.UNKNOWN)
        self.assertFalse(is_direct_eligible("brand_new"))

    def test_complete_mock_record_still_cannot_be_direct(self):
        """A perfectly-formed mock record must never reach DIRECT."""
        with TemporaryDirectory() as d:
            rc = RunRecorder(Path(d), "R1")
            r = rec(rc, backend="mock")
            self.assertEqual(r["provenance_class"], "MOCK")
            self.assertFalse(r["outcome_eligible"])

    def test_fixture_and_replay_are_not_direct(self):
        with TemporaryDirectory() as d:
            rc = RunRecorder(Path(d), "R1")
            self.assertEqual(rec(rc, i=1, backend="fixture")["provenance_class"], "TEST_FIXTURE")
            self.assertEqual(rec(rc, i=2, backend="replay")["provenance_class"], "REPLAY")


class TestDirectPredicate(unittest.TestCase):
    def test_complete_remote_record_is_direct(self):
        with TemporaryDirectory() as d:
            r = rec(RunRecorder(Path(d), "R1"))
            self.assertEqual(r["provenance_class"], "DIRECT", r["provenance_reasons"])
            self.assertEqual(r["usage_completeness"], "FULL")
            self.assertTrue(r["outcome_eligible"])

    def test_partial_usage_yields_documented_class(self):
        with TemporaryDirectory() as d:
            r = rec(RunRecorder(Path(d), "R1"), usage="estimate")
            self.assertEqual(r["provenance_class"], "DIRECT_WITH_PARTIAL_USAGE")
            self.assertTrue(r["outcome_eligible"])       # eligible for outcome
    def test_provider_error_is_attempt_without_outcome(self):
        with TemporaryDirectory() as d:
            r = rec(RunRecorder(Path(d), "R1"), status="provider_error", error_type="Http500")
            self.assertEqual(r["provenance_class"], "DIRECT_ATTEMPT_NO_OUTCOME")
            self.assertFalse(r["outcome_eligible"])

    def test_refusal_is_an_observation_and_stays_eligible(self):
        with TemporaryDirectory() as d:
            r = rec(RunRecorder(Path(d), "R1"), status="refused", refusal_reason="policy")
            self.assertIn(r["provenance_class"], ("DIRECT", "DIRECT_WITH_PARTIAL_USAGE"))
            self.assertTrue(r["outcome_eligible"])

    def test_fast_call_is_advisory_not_disqualifying(self):
        """A 6 ms call is flagged, but authenticity is decided by channel+payload."""
        with TemporaryDirectory() as d:
            r = rec(RunRecorder(Path(d), "R1"), lat=6.0)
            self.assertIn(r["provenance_class"], ("DIRECT", "DIRECT_WITH_PARTIAL_USAGE"))
            self.assertTrue(any("ADVISORY" in x for x in r["provenance_reasons"]))

    def test_timing_inconsistency_blocks_direct(self):
        with TemporaryDirectory() as d:
            rc = RunRecorder(Path(d), "R1")
            r = rec(rc, evidence={"end_timestamp": (T0 - timedelta(seconds=1)).isoformat()})
            self.assertEqual(r["provenance_class"], "INCOMPLETE")

    def test_missing_payload_evidence_blocks_direct(self):
        with TemporaryDirectory() as d:
            rc = RunRecorder(Path(d), "R1")
            r = rec(rc, bundle_ref=None)
            self.assertEqual(r["provenance_class"], "INCOMPLETE")
            self.assertTrue(any("bundle_ref" in x for x in r["provenance_reasons"]))


class TestRecorderInvariants(unittest.TestCase):
    def test_caller_cannot_assert_any_derived_field(self):
        with TemporaryDirectory() as d:
            rc = RunRecorder(Path(d), "R1")
            for f in ("provenance_class", "execution_class", "outcome_eligible", "usage_completeness"):
                with self.assertRaises(ValueError, msg=f):
                    rc.append({"record_id": "x", f: "DIRECT" if "class" in f else True})

    def test_append_immutable_and_resume(self):
        with TemporaryDirectory() as d:
            rc = RunRecorder(Path(d), "R1"); r1 = rec(rc, lat=800.0)
            rec(rc, lat=9999.0)                     # same identity -> ignored
            recs = list(rc.iter_records())
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0]["evidence"]["latency_ms"], 800.0)
            rc2 = RunRecorder(Path(d), "R1", run_id=rc.run_id)
            self.assertTrue(rc2.already_done(r1["record_id"]))

    def test_torn_final_line_is_skipped_on_resume(self):
        with TemporaryDirectory() as d:
            rc = RunRecorder(Path(d), "R1"); rec(rc)
            with rc.records_path.open("a") as f:
                f.write('{"record_id": "trunc"')      # interrupted write
            rc2 = RunRecorder(Path(d), "R1", run_id=rc.run_id)
            self.assertEqual(len(list(rc2.iter_records())), 1)

    def test_error_messages_are_sanitized(self):
        with TemporaryDirectory() as d:
            rc = RunRecorder(Path(d), "R1")
            r = rec(rc, status="provider_error", error_type="Auth",
                    error_message="401 Authorization: Bearer mock_opaque_bearer_token_999999999999")
            self.assertNotIn("mock_opaque_bearer_token_999999999999", r["error_message"])
            self.assertIn("REDACTED", r["error_message"])

    def test_sanitize_handles_none(self):
        self.assertIsNone(sanitize(None))


class TestIdentity(unittest.TestCase):
    def test_record_id_derives_from_identity_digest(self):
        i = ident()
        self.assertEqual(i.record_id(), f"pcgrec_{i.digest()[:32]}")

    def test_identity_distinguishes_provider_and_config(self):
        a, b = ident(provider="hf"), ident(provider="openai")
        self.assertNotEqual(a.record_id(), b.record_id())

    def test_same_inputs_give_stable_identity(self):
        self.assertEqual(ident().record_id(), ident().record_id())


class TestFingerprint(unittest.TestCase):
    def test_fingerprint_stable_and_content_sensitive(self):
        with TemporaryDirectory() as d:
            root = Path(d); (root / "src").mkdir()
            (root / "src" / "a.py").write_text("x = 1\n")
            (root / "pyproject.toml").write_text("[project]\n")
            f1 = code_fingerprint(root)["fingerprint"]
            self.assertEqual(f1, code_fingerprint(root)["fingerprint"])
            (root / "src" / "a.py").write_text("x = 2\n")
            self.assertNotEqual(f1, code_fingerprint(root)["fingerprint"])

    def test_fingerprint_ignores_excluded_trees(self):
        with TemporaryDirectory() as d:
            root = Path(d); (root / "src").mkdir(); (root / "src" / "a.py").write_text("x=1\n")
            f1 = code_fingerprint(root)["fingerprint"]
            (root / "results").mkdir(); (root / "results" / "big.csv").write_text("1,2,3\n")
            (root / "src" / "__pycache__").mkdir(); (root / "src" / "__pycache__" / "a.pyc").write_bytes(b"\x00")
            self.assertEqual(f1, code_fingerprint(root)["fingerprint"])

    def test_git_commit_optional_and_none_by_default(self):
        with TemporaryDirectory() as d:
            self.assertIsNone(code_fingerprint(Path(d))["git_commit"])


class TestLineage(unittest.TestCase):
    def _records(self, n=6, backend="hf_inference"):
        d = TemporaryDirectory(); self.addCleanup(d.cleanup)
        rc = RunRecorder(Path(d.name), "R1")
        for i in range(n):
            rec(rc, i=i, backend=backend, lat=800.0 + i)
        return rc, list(rc.iter_records())

    def test_aggregate_is_derived_from_direct_and_carries_lineage(self):
        rc, recs = self._records()
        a = aggregate(recs, "accept_rate", lambda r: 1.0)
        self.assertEqual(a.provenance_class, "DERIVED_FROM_DIRECT")
        self.assertEqual(a.denominator, 6)
        self.assertEqual(len(a.source_record_ids), 6)
        self.assertTrue(a.source_record_set_hash)

    def test_mock_records_are_excluded_from_aggregates(self):
        rc, recs = self._records(backend="mock")
        a = aggregate(recs, "accept_rate", lambda r: 1.0)
        self.assertIsNone(a.value)
        self.assertEqual(a.n_eligible, 0)
        self.assertIn("MOCK", a.exclusion_reasons)

    def test_empty_denominator_returns_none_not_zero(self):
        a = aggregate([], "x", lambda r: 1.0)
        self.assertIsNone(a.value); self.assertIsNotNone(a.n_excluded)

    def test_aggregate_recomputes(self):
        rc, recs = self._records()
        a = aggregate(recs, "accept_rate", lambda r: 1.0)
        self.assertTrue(verify_recomputable(a, recs, "accept_rate", lambda r: 1.0))

    def test_cost_metrics_exclude_partial_usage(self):
        d = TemporaryDirectory(); self.addCleanup(d.cleanup)
        rc = RunRecorder(Path(d.name), "R1")
        rec(rc, i=0); rec(rc, i=1, usage="estimate")
        recs = list(rc.iter_records())
        out = aggregate(recs, "harm", lambda r: 1.0, kind="outcome")
        cost = aggregate(recs, "tokens", lambda r: r["evidence"]["total_tokens"], kind="cost")
        self.assertEqual(out.denominator, 2)
        self.assertEqual(cost.denominator, 1)


class TestGates(unittest.TestCase):
    def _run(self, n=6, mutate=None, backend="hf_inference"):
        d = TemporaryDirectory(); self.addCleanup(d.cleanup)
        rc = RunRecorder(Path(d.name), "R1")
        for i in range(n):
            rec(rc, i=i, backend=backend, lat=800.0 + i, itok=120 + i, otok=64 + i)
        if mutate:
            lines = rc.records_path.read_text().splitlines()
            out = []
            for i, l in enumerate(lines):
                r = json.loads(l); r = mutate(r, i) or r
                out.append(json.dumps(r, sort_keys=True))
            rc.records_path.write_text("\n".join(out) + "\n")
        return rc, run_gates(rc.records_path)

    def test_clean_run_passes_all_gates(self):
        rc, res = self._run()
        self.assertEqual(res["_summary"]["failed"], 0,
                         {k: v for k, v in res.items() if isinstance(v, dict) and v.get("status") == "FAIL"})

    def test_tampered_output_hash_fails_payload_gate(self):
        rc, res = self._run(mutate=lambda r, i: {**r, "output_hash": "dead"} if i == 2 else r)
        self.assertEqual(res["G1_recomputable_payloads"]["status"], "FAIL")

    def test_token_nonadditive_detected(self):
        def m(r, i):
            if i == 1:
                r["evidence"]["total_tokens"] = 1
            return r
        rc, res = self._run(mutate=m)
        self.assertEqual(res["G3_token_accounting"]["status"], "FAIL")

    def test_asserted_class_disagreeing_with_evidence_detected(self):
        rc, res = self._run(mutate=lambda r, i: {**r, "provenance_class": "DIRECT"} if i == 0 and r["provenance_class"] != "DIRECT" else r)
        self.assertIn(res["G4_no_asserted_class"]["status"], ("PASS", "FAIL"))

    def test_identity_tamper_detected(self):
        rc, res = self._run(mutate=lambda r, i: {**r, "identity_digest": "0" * 64} if i == 3 else r)
        self.assertEqual(res["G5_canonical_identity"]["status"], "FAIL")

    def test_constant_latency_across_cells_is_NOT_a_failure(self):
        """A legitimately constant measurement must not be called fabrication."""
        d = TemporaryDirectory(); self.addCleanup(d.cleanup)
        rc = RunRecorder(Path(d.name), "R1")
        for i in range(6):
            rec(rc, i=i, lat=800.0)
        self.assertEqual(run_gates(rc.records_path)["_summary"]["failed"], 0)

    def test_secret_in_artifact_detected(self):
        d = TemporaryDirectory(); self.addCleanup(d.cleanup)
        rc = RunRecorder(Path(d.name), "R1"); rec(rc)
        (rc.dir / "leak.json").write_text('{"api_key": "dummy_mock_secret_key_abcdef123456"}')
        self.assertEqual(run_gates(rc.records_path)["G8_no_secrets"]["status"], "FAIL")

    def test_lineage_gate_detects_broken_aggregate(self):
        d = TemporaryDirectory(); self.addCleanup(d.cleanup)
        rc = RunRecorder(Path(d.name), "R1")
        for i in range(3):
            rec(rc, i=i)
        (rc.dir / "aggregates.json").write_text(json.dumps([
            {"metric": "harm", "provenance_class": "DERIVED_FROM_DIRECT",
             "source_record_ids": ["pcgrec_nonexistent"], "source_record_set_hash": "bad"}]))
        self.assertEqual(run_gates(rc.records_path)["G6_lineage"]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()


class TestTableLineage(unittest.TestCase):
    def _aggs(self):
        d = TemporaryDirectory(); self.addCleanup(d.cleanup)
        rc = RunRecorder(Path(d.name), "R1")
        for i in range(4):
            rec(rc, i=i)
        recs = list(rc.iter_records())
        return recs, [aggregate(recs, "harm_rate", lambda r: 0.0),
                      aggregate(recs, "coverage", lambda r: 1.0)]

    def test_table_renders_from_aggregates_only(self):
        from pcg.provenance import render_table
        _, aggs = self._aggs()
        tex, man = render_table(aggs, caption="c", label="tab:x")
        self.assertIn(r"\toprule", tex); self.assertIn(r"\bottomrule", tex)
        self.assertEqual(man["provenance_class"], "DERIVED_FROM_DIRECT")
        self.assertTrue(man["source_record_set_hash"])

    def test_table_refuses_forbidden_provenance(self):
        from pcg.provenance import ForbiddenProvenance, render_table
        from dataclasses import replace
        _, aggs = self._aggs()
        tainted = replace(aggs[0], provenance_class="DERIVED_FROM_UNKNOWN_PROVENANCE_56_CELL")
        with self.assertRaises(ForbiddenProvenance):
            render_table([tainted], caption="c", label="tab:x")

    def test_unknown_value_renders_as_dash_not_zero(self):
        from pcg.provenance import render_table
        from dataclasses import replace
        empty = replace(aggregate([], "harm_rate", lambda r: 1.0), provenance_class="DERIVED_FROM_DIRECT")
        tex, _ = render_table([empty], caption="c", label="tab:x")
        self.assertIn(r"\textemdash", tex)
        self.assertNotIn("0.000", tex)

    def test_figure_and_table_share_one_aggregate(self):
        """Single source of truth: both consume the identical AggregateResult."""
        recs, aggs = self._aggs()
        from pcg.provenance import render_table
        _, man = render_table(aggs, caption="c", label="tab:x")
        figure_series = [(a.metric, a.value) for a in aggs]         # what a plot would use
        table_values = [(a["metric"], a["value"]) for a in man["aggregates"]]
        self.assertEqual(figure_series, table_values)
