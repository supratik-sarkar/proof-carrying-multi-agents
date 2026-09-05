#!/usr/bin/env python3
"""Deterministic offline fixtures for A01-A18. No model or network call."""
from __future__ import annotations
import json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from pcg.v3.record import PerExampleRecord
from pcg.v3.canon import sha256_file

SEED = 20270301
BASE = os.path.join("tests", "fixtures", "v3")
CELLS = ["phi-3.5-mini/FEVER", "qwen2.5-7B/HotpotQA", "Llama-3.1-8B/PubMedQA",
         "Gemma-2-9b-it/TAT-QA", "Llama-3.3-70B/ToolBench", "deepseek-v3/WebLINX"]
SYSTEMS = ["nocert", "citation_only", "shieldagent", "agentrr", "pcg_mas"]

def rec(i, eid, system, rng, **over):
    d = dict(record_id=f"{eid}-{system}-{i:05d}", run_id=f"FIXTURE-{eid}", experiment_id=eid,
             system=system, provenance_class="TEST_FIXTURE",
             cell_id=CELLS[i % len(CELLS)], dataset="fixture", split="fixture",
             example_id=f"ex{i:05d}", seed=i % 4,
             model_id="fixture/deterministic", model_revision="0", tokenizer_id="fixture",
             backend_type="MOCK", provider_route="offline_mock", dtype="float32",
             decoding_config_hash="d0", prompt_hash="p0",
             backend_fingerprint=f"fp-{CELLS[i % len(CELLS)]}",
             checker_fingerprint="checker-v3-fixture",
             v_h=True, v_pi=True, v_gamma=True, v_entail=True, check=True,
             int_fail=False, replay_fail=False, drift_fail=False, check_fail=False,
             cov_gap=False, n_channels_fired=0, contract_bad=False,
             attempted=True, answered=True, accepted=True, controller_action="Answer",
             h_support=0.0, h_exec=0.0, h_joint=0.0, utility=round(rng.random(), 4),
             cov_cert=round(rng.random(), 4), cov_audit=round(rng.random(), 4),
             latency_ms=round(80 + 60 * rng.random(), 2), tokens_in=120, tokens_out=45,
             model_calls=1, retrieval_calls=1, tool_calls=0, checker_calls=1, replay_calls=0,
             billed_cost_usd=0.0, stratum_id="h0", sampling_weight=0.25,
             host_fingerprint="host-A", device="cpu")
    d.update(over)
    return d

def write(eid, rows, note):
    d = os.path.join(BASE, eid.lower()); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "records.jsonl")
    with open(p, "w") as fh:
        for r in rows:
            errs = PerExampleRecord(**{k: v for k, v in r.items()
                                       if k in PerExampleRecord.__dataclass_fields__}).validate()
            if errs:
                raise SystemExit(f"{eid} fixture invalid: {errs} in {r['record_id']}")
            fh.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
    open(os.path.join(d, "README.md"), "w").write(
        f"# Fixture {eid}\n\n{note}\n\nRows: {len(rows)}  \nSHA-256: `{sha256_file(p)}`  \n"
        f"Deterministic (seed {SEED}); no model or network call. `provenance_class = TEST_FIXTURE` "
        "throughout: these records may never be promoted to DIRECT evidence.\n")
    return len(rows)

def main():
    rng = random.Random(SEED); total = 0
    total += write("A01", [rec(i, "A01", "pcg_mas", rng) for i in range(120)],
                   "One backend fingerprint per cell across four seeds.")
    rows = []
    for i in range(600):
        s = SYSTEMS[i % 5]; acc = rng.random() < (0.9 if s == "pcg_mas" else 0.85)
        harm = acc and rng.random() < (0.05 if s == "pcg_mas" else 0.12)
        rows.append(rec(i, "A02", s, rng, accepted=acc, answered=acc, h_joint=1.0 if harm else 0.0,
                        controller_action="Answer" if acc else "Refuse"))
    total += write("A02", rows, "Cross-system accepted/harm counts; denominators explicit.")
    rows = []
    for i in range(800):
        ans = i % 5 != 0; lnc = round(rng.random(), 6)
        rows.append(rec(i, "A03", "pcg_mas", rng, answered=ans, accepted=ans, loss_nocert=lnc,
                        loss_pcg=(round(lnc * rng.random() * 0.8, 6) if ans else None),
                        controller_action="Answer" if ans else "Refuse"))
    total += write("A03", rows, "Per-example losses so Delta = S + V is assertable to 1e-12.")
    rows = []
    for i in range(200):
        for host in ("host-A", "host-B"):
            flip = (i in (37, 118)) and host == "host-B"
            rows.append(rec(i, "A04", "pcg_mas", rng, host_fingerprint=host,
                            record_id=f"A04-cert-{i:05d}", check=not flip,
                            v_entail=not flip, accepted=not flip))
    total += write("A04", rows, "Cross-host recomputation; two disagreements deliberately present and disclosed.")
    rows = []
    for i in range(1000):
        s = SYSTEMS[i % 5]; acc = rng.random() < 0.85
        rows.append(rec(i, "A05", s, rng, accepted=acc, answered=acc,
                        h_support=1.0 if (acc and rng.random() < (0.03 if s == "pcg_mas" else 0.10)) else 0.0,
                        h_exec=1.0 if (acc and rng.random() < (0.02 if s == "pcg_mas" else 0.08)) else 0.0,
                        h_joint=1.0 if (acc and rng.random() < (0.04 if s == "pcg_mas" else 0.13)) else 0.0))
    total += write("A05", rows, "Five systems, matched coverage target, native-scope harm split.")
    rows = []; fams = [("W_H", "v_h"), ("W_Pi", "v_pi"), ("W_Gamma", "v_gamma"), ("W_vdash", "v_entail")]
    for fi, (fam, key) in enumerate(fams):
        for i in range(110):
            o = {k: True for _, k in fams}; o[key] = False
            rows.append(rec(fi * 110 + i, "A06", "pcg_mas", rng, stratum_id=fam,
                            check=False, accepted=False, **o))
    total += write("A06", rows, "110 instances per family; exactly one conjunct False, other three True.")
    rows = []
    for i in range(1200):
        bad = rng.random() < 0.06
        fired = {k: False for k in ("int_fail","replay_fail","drift_fail","check_fail","cov_gap")}
        if bad: fired[rng.choice(list(fired))] = True
        rows.append(rec(i, "A07", "pcg_mas", rng, stratum_id=f"h{i % 4}", sampling_weight=0.2,
                        contract_bad=bad, n_channels_fired=sum(fired.values()), **fired))
    total += write("A07", rows, "Four covered strata at mass 0.2 each; pi_unc = 0.2 charged once.")
    rows = []
    for regime, q in (("isolated_verifier", 0.0), ("shared_verifier", 0.06),
                      ("poisoned_shard", 0.05), ("malicious_tool_output", 0.04)):
        for i in range(400):
            bf = [True]*4 if rng.random() < q else [rng.random() < 0.18 for _ in range(4)]
            acc = all(bf)
            rows.append(rec(i, "A08", "pcg_mas", rng, injection_regime=regime, k_redundancy=4,
                            branch_failures=bf, accepted=acc, answered=True,
                            h_joint=1.0 if acc else 0.0, contract_bad=acc))
    total += write("A08", rows, "Four attack regimes; shared-cause mass injected in three of them.")
    rows = []
    for regime in ("none","held_out_dataset","backend_change","corruption","tool_drift"):
        for i in range(300):
            rows.append(rec(i, "A09", "pcg_mas", rng, shift_regime=regime,
                            contract_bad=rng.random() < (0.04 if regime == "none" else 0.11)))
    total += write("A09", rows, "Five deployment-shift regimes for the alarm sweep.")
    rows = []
    for i in range(1000):
        s = SYSTEMS[i % 5]
        mult = 1.0 if s == "nocert" else (1.25 if s in ("citation_only","shieldagent") else 1.75)
        rows.append(rec(i, "A10", s, rng, latency_ms=round(90*mult*(0.8+0.6*rng.random()), 2),
                        tokens_in=int(120*mult), tokens_out=int(45*mult),
                        model_calls=1 if s != "pcg_mas" else 3,
                        checker_calls=0 if s == "nocert" else (1 if s != "pcg_mas" else 2),
                        retrieval_calls=1 if s != "nocert" else 0,
                        replay_calls=0 if s != "pcg_mas" else 1,
                        billed_cost_usd=round(0.0004*mult, 6), accepted=rng.random() < 0.85,
                        h_joint=1.0 if rng.random() < 0.06 else 0.0))
    total += write("A10", rows, "Per-record cost telemetry; A10 aggregates, never re-runs.")
    rows = []
    for i in range(900):
        keys = ["int_fail","replay_fail","drift_fail","check_fail","cov_gap"]
        nf = rng.choices([0,1,2,3], weights=[62,24,10,4])[0]
        on = rng.sample(keys, nf); fired = {k: (k in on) for k in keys}
        rows.append(rec(i, "A11", "pcg_mas", rng, n_channels_fired=nf, contract_bad=nf > 0,
                        check=nf == 0, **fired))
    total += write("A11", rows, "Channel-firing multiplicities for Lambda_union = E[(N_F-1)_+].")
    rows = []
    for i in range(1500):
        bf = [True]*4 if rng.random() < 0.05 else [rng.random() < 0.22 for _ in range(4)]
        rows.append(rec(i, "A12", "pcg_mas", rng, k_redundancy=4, branch_failures=bf,
                        accepted=all(bf), contract_bad=all(bf)))
    total += write("A12", rows, "Four-branch failure vectors; 2^k table populated for D_inf.")
    rows = []; comps = ["hash","canon","log_schema","clock","version_pin","replay_runner","entail_tcb","exec_tcb"]
    for ci, comp in enumerate(comps):
        for j in range(40):
            flip = j < (4 + ci % 5)
            rf = bool(flip and comp == "replay_runner")
            rows.append(rec(ci*40+j, "A13", "pcg_mas", rng, stratum_id=comp, check=not flip,
                            accepted=not flip, contract_bad=flip,
                            v_h=not (flip and comp in ("hash","canon")),
                            replay_fail=rf, n_channels_fired=int(rf)))
    total += write("A13", rows, "Eight trusted-base components perturbed; per-component flip rates.")
    rows = []; fams2 = ["memory_poisoning","corrupted_ocr","stale_api_state","coordinated_deception","schema_bypass","compromised_delegation"]
    for fi, fam in enumerate(fams2):
        for i in range(150):
            bad = rng.random() < 0.20
            keys = ["int_fail","replay_fail","drift_fail","check_fail","cov_gap"]
            fired = {k: False for k in keys}
            if bad:
                u = rng.random()
                if u < 0.12: pass
                elif u < 0.30: fired["cov_gap"] = True
                else: fired[rng.choice(keys[:4])] = True
            nf = sum(1 for v in fired.values() if v)
            rows.append(rec(fi*150+i, "A14", "pcg_mas", rng, corruption=fam, contract_bad=bad,
                            n_channels_fired=nf, eps_tax_label=(bad and nf == 0), **fired))
    total += write("A14", rows, "Six held-out attack families; some ContractBad events caught by no channel.")
    rows = []
    for cond, fpr in (("nominal",0.04),("weaker_checker",0.14),("context_truncation",0.11),
                      ("distractor_evidence",0.09),("negation_traps",0.17)):
        for i in range(300):
            src_bad = rng.random() < 0.25
            ent = (rng.random() > 0.05) if not src_bad else (rng.random() < fpr)
            cf = bool(src_bad and ent)
            rows.append(rec(i, "A15", "pcg_mas", rng, corruption=cond, v_entail=ent, check=ent,
                            accepted=ent, eps_src_label=src_bad, check_fail=cf,
                            n_channels_fired=int(cf), contract_bad=cf,
                            h_joint=1.0 if (ent and src_bad) else 0.0,
                            checker_fingerprint="checker-v3-fixture"))
    total += write("A15", rows, "Nominal plus four pre-registered degraded checker conditions.")
    rows = []
    for arm in ("equal_model_calls","equal_tokens","equal_cost","allocation_matched"):
        for i in range(400):
            s = SYSTEMS[i % 5]; is_pcg = s == "pcg_mas"
            rows.append(rec(i, "A16", s, rng, cell_id=arm, model_calls=3,
                            checker_calls=2 if is_pcg else 0,
                            retrieval_calls=2 if is_pcg else 3, tokens_in=140, tokens_out=50,
                            billed_cost_usd=0.0009, accepted=rng.random() < 0.86,
                            h_joint=1.0 if rng.random() < (0.05 if is_pcg else 0.10) else 0.0))
    total += write("A16", rows, "Four equal-budget arms including an allocation-matched arm.")
    rows = []
    for i in range(800):
        r = rng.random()
        rows.append(rec(i, "A17", "pcg_mas", rng, cov_cert=round(r, 6),
                        utility=round(max(0.0, min(1.0, r + rng.gauss(0, 0.05))), 6)))
    total += write("A17", rows, "Paired true/estimated risk for A17-A and A17-B.")
    total += write("A18", [rec(i, "A18", "pcg_mas", rng, provenance_class="MODELLED") for i in range(40)],
                   "OPTIONAL. Privacy is MODELLED in v3.0; A18 is not a blocker.")
    print(f"fixtures written: {total} rows across 18 workstreams")

if __name__ == "__main__":
    main()
