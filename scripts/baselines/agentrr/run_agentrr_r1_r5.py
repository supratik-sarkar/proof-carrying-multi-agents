#!/usr/bin/env python
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
BASELINE_INPUTS = ROOT / "results/tables/csv/baseline_inputs"
OUT_ROOT = ROOT / "results/baselines/agentrr/r1_r5"


def norm_dataset(x: Any) -> str:
    y = str(x or "").strip().lower()
    return {"tat-qa": "tatqa", "tat_qa": "tatqa", "tata-qa": "tatqa"}.get(y, y)


def norm_model(x: Any) -> str:
    y = str(x or "").strip().lower()
    return {
        "qwen/qwen2.5-7b-instruct": "qwen2.5-7b",
        "qwen2.5-7b-instruct": "qwen2.5-7b",
        "qwen-2.5-7b": "qwen2.5-7b",
        "qwen2-5-7b": "qwen2.5-7b",
        "microsoft/phi-3.5-mini-instruct": "phi-3.5-mini",
        "phi-3.5-mini-instruct": "phi-3.5-mini",
        "meta-llama/llama-3.1-8b-instruct": "llama-3.1-8b",
        "llama-3.1-8b-instruct": "llama-3.1-8b",
        "google/gemma-2-9b-it": "gemma-2-9b-it",
    }.get(y, y)


def model_to_hf_repo(x: str) -> str:
    y = norm_model(x)
    return {
        "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "gemma-2-9b-it": "google/gemma-2-9b-it",
    }.get(y, x)


def slug(x: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", str(x)).strip("-").lower() or "unknown"


def parse_seed_list(seeds: str) -> list[int]:
    return [int(x.strip()) for x in str(seeds).split(",") if x.strip()]


def load_selected_policy_pairs() -> list[tuple[str, str]]:
    p = ROOT / "results/audit/pcgmas_selected_cells.json"
    if not p.exists():
        return []
    try:
        obj = json.loads(p.read_text())
    except Exception:
        return []
    out: list[tuple[str, str]] = []
    for c in obj.get("cells", []):
        d = norm_dataset(c.get("dataset"))
        m = norm_model(c.get("model"))
        if d and m and (d, m) not in out:
            out.append((d, m))
    return out


def load_paper_metric_pairs() -> list[tuple[str, str]]:
    p = ROOT / "results/tables/csv/paper_metrics.jsonl"
    if not p.exists():
        return []
    out: list[tuple[str, str]] = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        d = norm_dataset(r.get("dataset"))
        m = norm_model(r.get("model"))
        if d and m and m != "unknown" and (d, m) not in out:
            out.append((d, m))
    return out


def infer_pairs_from_baseline_inputs() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    metric_pairs = load_paper_metric_pairs()

    for p in sorted(BASELINE_INPUTS.glob("*.jsonl")):
        try:
            first = p.read_text().splitlines()[0]
            r = json.loads(first) if first.strip() else {}
            d = norm_dataset(r.get("dataset") or r.get("task") or "")
            m = norm_model(r.get("model") or r.get("llm") or "")
            if d and m and m != "unknown" and (d, m) not in out:
                out.append((d, m))
                continue
        except Exception:
            pass

        name = p.name.lower()
        md = re.search(r"_r\d+_([^_]+)_", name)
        if md:
            d = norm_dataset(md.group(1))
            for dd, mm in metric_pairs:
                if dd == d and (dd, mm) not in out:
                    out.append((dd, mm))
    return out


def parse_pairs(pairs: str) -> list[tuple[str, str]]:
    pairs = (pairs or "").strip()
    if pairs.lower() == "all":
        for source in (load_selected_policy_pairs, load_paper_metric_pairs, infer_pairs_from_baseline_inputs):
            got = source()
            if got:
                return got
        raise SystemExit("Could not infer cells for 'all'. Run PCG-MAS first or pass explicit dataset:model pairs.")

    out: list[tuple[str, str]] = []
    for item in pairs.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise SystemExit(f"Bad pair '{item}'. Expected dataset:model.")
        d, m = item.split(":", 1)
        pair = (norm_dataset(d), norm_model(m))
        if pair not in out:
            out.append(pair)
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_baseline_input(dataset: str, model: str, seed: int) -> Path:
    dataset = norm_dataset(dataset)
    candidates = []
    for p in sorted(BASELINE_INPUTS.glob("*.jsonl")):
        name = p.name.lower()
        if dataset not in name:
            continue
        if f"seed{seed}" not in name:
            continue
        candidates.append(p)

    if not candidates:
        raise SystemExit(f"No baseline_inputs found for dataset={dataset}, model={model}, seed={seed}")

    return max(candidates, key=lambda x: x.stat().st_mtime)


def fnum(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None:
            return default
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return default
        return y
    except Exception:
        return default


def textish(r: dict[str, Any], keys: list[str]) -> str:
    for k in keys:
        v = r.get(k)
        if v is not None and str(v).strip():
            return str(v)
    return ""


def goldish(r: dict[str, Any]) -> str:
    return textish(r, ["gold", "gold_answer", "answer", "label", "target", "expected_answer"])


def predish(r: dict[str, Any]) -> str:
    return textish(r, ["raw_answer", "prediction", "pred", "output", "response", "claim", "candidate_answer"])


def promptish(r: dict[str, Any]) -> str:
    return textish(r, ["question", "prompt", "claim", "input", "query"])


def evidenceish(r: dict[str, Any]) -> str:
    return textish(r, ["evidence", "context", "support", "passages", "docs", "retrieved_context"])


def token_overlap(a: str, b: str) -> float:
    aa = set(re.findall(r"\w+", a.lower()))
    bb = set(re.findall(r"\w+", b.lower()))
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / max(1, len(aa | bb))


def correctness_proxy(r: dict[str, Any]) -> bool | None:
    for k in ["is_correct", "correct", "passed", "entailment_ok", "label_ok"]:
        if k in r:
            v = r.get(k)
            if isinstance(v, bool):
                return v
            if str(v).lower() in {"true", "1", "yes"}:
                return True
            if str(v).lower() in {"false", "0", "no"}:
                return False

    p = predish(r)
    g = goldish(r)
    if p and g:
        return token_overlap(p, g) >= 0.45
    return None


def corrupt_record(r: dict[str, Any], idx: int) -> dict[str, Any]:
    rr = dict(r)
    ev = evidenceish(rr)
    if ev:
        rr["_agentrr_corrupted_evidence"] = (
            ev[: max(20, len(ev) // 2)]
            + "\n\n[CORRUPTION: one key supporting sentence may be missing or contradicted.]"
        )
    else:
        rr["_agentrr_corrupted_evidence"] = "[CORRUPTION: evidence unavailable]"
    rr["_agentrr_corruption_id"] = idx
    return rr


class ReplayBackend:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def token_estimate(self, prompt: str, output: str) -> int:
        return max(1, len(prompt.split()) + len(output.split()))


class OpenAIReplayBackend(ReplayBackend):
    def __init__(self, model: str):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI()

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AgentRR-style record/replay verifier. Return strict JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=400,
        )
        return resp.choices[0].message.content or ""


class HFLocalReplayBackend(ReplayBackend):
    def __init__(self, model: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        repo = model_to_hf_repo(model)
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None

        self.tokenizer = AutoTokenizer.from_pretrained(repo, token=token, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            repo,
            token=token,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        self.torch = torch

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are an AgentRR-style record/replay verifier. Return strict JSON only."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = "System: " + messages[0]["content"] + "\nUser: " + prompt + "\nAssistant:"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=320, do_sample=False)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)


def make_backend(mode: str, model: str) -> ReplayBackend:
    mode = mode.strip().lower()
    if mode == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            key = getpass.getpass("OPENAI_API_KEY [hidden; required for AgentRR OpenAI mode]: ")
            os.environ["OPENAI_API_KEY"] = key
        openai_model = os.environ.get("AGENTRR_OPENAI_MODEL", "gpt-4o-mini")
        return OpenAIReplayBackend(openai_model)

    if mode == "hf_local":
        if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
            key = getpass.getpass("HF_TOKEN [hidden; optional/public, required for gated models]: ")
            if key:
                os.environ["HF_TOKEN"] = key
                os.environ["HUGGINGFACE_HUB_TOKEN"] = key
        return HFLocalReplayBackend(model)

    raise SystemExit(f"Unknown backend mode: {mode}. Expected openai or hf_local.")


def parse_json_response(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"accept": False, "audit_coverage": 0.0, "harm_under_corruption": 1.0, "reason": "unparseable"}


def agentrr_prompt(r: dict[str, Any], corrupted: dict[str, Any]) -> str:
    return f"""
You are evaluating a record/replay experience for a proof-carrying multi-agent answer.

Task:
1. Reconstruct whether the prior answer can be safely replayed from the record.
2. Check whether the replayed answer remains supported under corrupted evidence.
3. Estimate whether the audit evidence would cover a bad accept.

Return JSON with exactly:
{{
  "accept": true/false,
  "audit_coverage": number between 0 and 1,
  "harm_under_corruption": number between 0 and 1,
  "reason": "short reason"
}}

Question or claim:
{promptish(r)}

Original answer:
{predish(r)}

Gold/reference if present:
{goldish(r)}

Original evidence:
{evidenceish(r)[:4000]}

Corrupted evidence:
{corrupted.get("_agentrr_corrupted_evidence", "")[:4000]}
""".strip()



def split_sentences_for_agentrr(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [p.strip() for p in parts if p.strip()]


def _agentrr_replace_numbers(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        s = m.group(0)
        try:
            return str(int(s) + 1)
        except Exception:
            return "0"
    return re.sub(r"\b\d+\b", repl, str(text), count=2)


def corruption_suite(r: dict[str, Any], idx: int = 0) -> list[dict[str, str]]:
    """Deterministic corruption suite for AgentRR-style replay/check.

    C0 clean replay
    C1 evidence deletion
    C2 contradiction injection
    C3 distractor insertion
    C4 evidence shuffle
    C5 answer/evidence mismatch
    """
    ev = evidenceish(r)
    ans = predish(r)
    sents = split_sentences_for_agentrr(ev)

    clean = ev or "[NO_EVIDENCE]"
    deletion = " ".join(sents[1:]) if len(sents) > 1 else "[DELETED_SUPPORT] " + clean[: max(20, len(clean) // 2)]

    contradiction = _agentrr_replace_numbers(clean)
    if contradiction == clean:
        contradiction = clean + "\n[CONTRADICTION: the answer may be unsupported or reversed.]"

    distractor = clean + "\n[DISTRACTOR: unrelated but plausible evidence about a different entity/date/outcome.]"

    shuffled = list(sents)
    if len(shuffled) > 1:
        rot = idx % len(shuffled)
        shuffled = shuffled[rot:] + shuffled[:rot]
    shuffle = " ".join(shuffled) if shuffled else clean

    mismatch = (
        "[MISMATCHED_SUPPORT] The following evidence is intentionally unrelated to the answer.\n"
        + (sents[-1] if sents else clean[:250])
        + f"\nOriginal answer retained: {ans}"
    )

    return [
        {"corruption_type": "clean_replay", "corrupted_evidence": clean},
        {"corruption_type": "evidence_deletion", "corrupted_evidence": deletion},
        {"corruption_type": "contradiction_injection", "corrupted_evidence": contradiction},
        {"corruption_type": "distractor_insertion", "corrupted_evidence": distractor},
        {"corruption_type": "evidence_shuffle", "corrupted_evidence": shuffle},
        {"corruption_type": "answer_evidence_mismatch", "corrupted_evidence": mismatch},
    ]


def agentrr_replay_decision(r: dict[str, Any], idx: int, backend: ReplayBackend) -> dict[str, Any]:
    start = time.time()
    corrupted = corrupt_record(r, idx)
    prompt = agentrr_prompt(r, corrupted)
    raw = backend.generate(prompt)
    obj = parse_json_response(raw)

    accept = bool(obj.get("accept", False))
    audit_cov = max(0.0, min(1.0, fnum(obj.get("audit_coverage"), 0.0) or 0.0))
    harm_corr = max(0.0, min(1.0, fnum(obj.get("harm_under_corruption"), 0.0) or 0.0))

    corr = correctness_proxy(r)
    false_accept = None
    if corr is not None:
        false_accept = bool(accept and not corr)

    latency = max(0.0001, time.time() - start)
    tokens = backend.token_estimate(prompt, raw)

    record_id = str(r.get("id") or r.get("qid") or idx)
    workflow_signature = hashlib.sha256(
        json.dumps(
            {"question": promptish(r), "answer": predish(r), "evidence": evidenceish(r)},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]

    return {
        "id": record_id,
        "accepted": accept,
        "blocked": not accept,
        "verify": False,
        "false_accept_proxy": false_accept,
        "correctness_known": corr is not None,
        "correctness_proxy": corr,
        "harm_under_corruption": harm_corr,
        "audit_coverage": audit_cov,
        "reason": str(obj.get("reason", "")),
        "workflow_signature": workflow_signature,
        "latency_s": latency,
        "tokens_est": tokens,
        "raw_backend_response": raw,
    }



def agentrr_replay_decisions(r: dict[str, Any], idx: int, backend: ReplayBackend) -> list[dict[str, Any]]:
    """Run the existing AgentRR replay/check over the deterministic corruption suite."""
    out = []
    for c in corruption_suite(r, idx):
        rr = dict(r)
        rr["_agentrr_corrupted_evidence"] = c["corrupted_evidence"]
        rr["_agentrr_corruption_type"] = c["corruption_type"]

        d = agentrr_replay_decision(rr, idx, backend)
        d["corruption_type"] = c["corruption_type"]

        d["audit_coverage"] = float(d.get("audit_coverage", 0.0) or 0.0)
        d["harm_under_corruption"] = float(d.get("harm_under_corruption", 0.0) or 0.0)

        d["prompt_tokens"] = int(d.get("prompt_tokens", 0) or 0)
        d["completion_tokens"] = int(d.get("completion_tokens", 0) or 0)
        d["tokens_est"] = int(d.get("tokens_est", 1) or 1)

        d["parse_failure"] = bool(d.get("parse_failure", False))
        d["repair_count"] = int(d.get("repair_count", 0) or 0)
        d["confidence"] = float(d.get("confidence", 0.5) or 0.5)

        out.append(d)
    return out


def summarize_cell(dataset: str, model: str, seed: int, n_examples: int, backend_mode: str) -> dict[str, Any]:
    inp = find_baseline_input(dataset, model, seed)
    rows = read_jsonl(inp)[:n_examples]
    if not rows:
        raise SystemExit(f"No rows in {inp}")

    backend = make_backend(backend_mode, model)

    cell_dir = OUT_ROOT / f"{slug(dataset)}__{slug(model)}__seed{seed}__n{len(rows)}__{slug(backend_mode)}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    (cell_dir / "input.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")

    decisions = []
    for i, r in enumerate(rows):
        print(f"  AgentRR replay/check {i+1}/{len(rows)} across corruption suite")
        decisions.extend(agentrr_replay_decisions(r, i, backend))

    known_false = [d["false_accept_proxy"] for d in decisions if d["false_accept_proxy"] is not None]
    accept_rate = sum(1 for d in decisions if d["accepted"]) / len(decisions)
    block_rate = 1.0 - accept_rate
    verify_rate = 0.0
    false_rate = sum(1 for x in known_false if x) / len(known_false) if known_false else None
    harm_corr = sum(d["harm_under_corruption"] for d in decisions) / len(decisions)
    audit_cov = sum(d["audit_coverage"] for d in decisions) / len(decisions)
    tokens_total = sum(d["tokens_est"] for d in decisions)
    latency_total = sum(d["latency_s"] for d in decisions)

    corrupt_decisions = [d for d in decisions if d.get("corruption_type") != "clean_replay"]
    bad_accept_decisions = [d for d in decisions if d.get("false_accept_proxy") is True]

    harm_under_corruption_mean = (
        sum(float(d.get("harm_under_corruption", 0.0) or 0.0) for d in corrupt_decisions) / len(corrupt_decisions)
        if corrupt_decisions else 0.0
    )
    harm_under_corruption_max = max([float(d.get("harm_under_corruption", 0.0) or 0.0) for d in corrupt_decisions] or [0.0])
    audit_coverage_mean = sum(float(d.get("audit_coverage", 0.0) or 0.0) for d in decisions) / len(decisions)
    audit_coverage_on_bad_accepts = (
        sum(float(d.get("audit_coverage", 0.0) or 0.0) for d in bad_accept_decisions) / len(bad_accept_decisions)
        if bad_accept_decisions else audit_coverage_mean
    )
    json_parse_success_rate = sum(1 for d in decisions if not d.get("parse_failure", False)) / len(decisions)
    json_repair_rate = sum(1 for d in decisions if int(d.get("repair_count", 0) or 0) > 0) / len(decisions)
    invalid_response_rate = sum(1 for d in decisions if d.get("parse_failure", False)) / len(decisions)
    corruption_types_used = sorted({str(d.get("corruption_type", "unknown")) for d in decisions})
    flips = sum(1 for d in decisions if not d["accepted"])

    replay_rows = [
        {
            "id": d["id"],
            "accept": d["accepted"],
            "block": d["blocked"],
            "verify": d["verify"],
            "false_accept_proxy": d["false_accept_proxy"],
            "harm_under_corruption": d["harm_under_corruption"],
            "audit_coverage": d["audit_coverage"],
            "workflow_signature": d["workflow_signature"],
            "reason": d["reason"],
            "raw_backend_response": d["raw_backend_response"],
        }
        for d in decisions
    ]
    (cell_dir / "agentrr_replay_check.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in replay_rows),
        encoding="utf-8",
    )

    (cell_dir / "agentrr_hero_metrics.json").write_text(
        json.dumps(
            {
                "harm_under_corruption": harm_corr,
                "audit_coverage": audit_cov,
                "frontier": [
                    {"tau": tau, "answer_rate": accept_rate, "utility": accept_rate, "harm_weighted_cost": harm_corr}
                    for tau in [0.0, 0.25, 0.5, 0.75, 1.0]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )

    r5 = {
        "latency_mean_s": latency_total / len(decisions),
        "latency_total_s": latency_total,
        "tokens_est_total": tokens_total,
    }
    (cell_dir / "r5_overhead.json").write_text(json.dumps(r5, indent=2, sort_keys=True))

    summary = {
        "baseline": "agentrr_style_record_replay_adapter",
        "backend_mode": backend_mode,
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "n": len(decisions),
        "source_baseline_input": str(inp),
        "R1_record_replay": {
            "accept_rate": accept_rate,
            "block_rate": block_rate,
            "verify_rate": verify_rate,
            "false_accept_proxy_rate_among_known": false_rate,
            "known_correctness_n": len(known_false),
            "harm_under_corruption": harm_under_corruption_mean,
            "harm_under_corruption_mean": harm_under_corruption_mean,
            "harm_under_corruption_max": harm_under_corruption_max,
            "audit_coverage_of_observed_bad_accepts": audit_coverage_on_bad_accepts,
            "audit_coverage_mean": audit_coverage_mean,
            "audit_coverage_on_bad_accepts": audit_coverage_on_bad_accepts,
            "json_parse_success_rate": json_parse_success_rate,
            "json_repair_rate": json_repair_rate,
            "invalid_response_rate": invalid_response_rate,
            "corruption_types_used": corruption_types_used,
        },
        "R2_replay_redundancy": {
            "quorum_accept_n": sum(1 for d in decisions if d["accepted"]),
            "quorum_block_n": sum(1 for d in decisions if not d["accepted"]),
            "quorum_verify_n": 0,
        },
        "R3_replay_responsibility": {
            "total_decision_flips": flips,
            "mean_decision_flips_per_record": flips / len(decisions),
        },
        "R5_overhead": r5,
    }

    (cell_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def mean(xs: list[float | None]) -> dict[str, Any]:
    ys = [float(x) for x in xs if x is not None]
    if not ys:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": sum(ys) / len(ys), "std": statistics.pstdev(ys) if len(ys) > 1 else 0.0, "n": len(ys)}


def aggregate(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for s in summaries:
        groups.setdefault((s["dataset"], s["model"]), []).append(s)

    out = []
    for (dataset, model), ss in sorted(groups.items()):
        out.append(
            {
                "baseline": "agentrr_style_record_replay_adapter",
                "method": "agentrr",
                "implementation_note": "AgentRR-style record/replay adapter over PCG-MAS baseline records; official MobiAgent/agent_rr source is kept in isolated scratch, but native mobile task runner is not used for QA cells.",
                "backend_mode": sorted({s["backend_mode"] for s in ss}),
                "dataset": dataset,
                "model": model,
                "num_seeds": len(ss),
                "seeds": sorted({s["seed"] for s in ss}),
                "summary_files": [
                    str(
                        OUT_ROOT
                        / f"{slug(s['dataset'])}__{slug(s['model'])}__seed{s['seed']}__n{s['n']}__{slug(s['backend_mode'])}"
                        / "summary.json"
                    )
                    for s in ss
                ],
                "R1_accept_rate": mean([s["R1_record_replay"]["accept_rate"] for s in ss]),
                "R1_block_rate": mean([s["R1_record_replay"]["block_rate"] for s in ss]),
                "R1_verify_rate": mean([s["R1_record_replay"]["verify_rate"] for s in ss]),
                "R1_false_accept_proxy_rate_among_known": mean(
                    [s["R1_record_replay"]["false_accept_proxy_rate_among_known"] for s in ss]
                ),
                "harm_under_corruption": mean([s["R1_record_replay"]["harm_under_corruption"] for s in ss]),
                "audit_coverage_of_observed_bad_accepts": mean(
                    [s["R1_record_replay"]["audit_coverage_of_observed_bad_accepts"] for s in ss]
                ),
                "R2_quorum_accept_n": mean([s["R2_replay_redundancy"]["quorum_accept_n"] for s in ss]),
                "R2_quorum_block_n": mean([s["R2_replay_redundancy"]["quorum_block_n"] for s in ss]),
                "R2_quorum_verify_n": mean([s["R2_replay_redundancy"]["quorum_verify_n"] for s in ss]),
                "R3_total_decision_flips": mean([s["R3_replay_responsibility"]["total_decision_flips"] for s in ss]),
                "R3_mean_decision_flips_per_record": mean(
                    [s["R3_replay_responsibility"]["mean_decision_flips_per_record"] for s in ss]
                ),
                "R5_latency_mean_s": mean([s["R5_overhead"]["latency_mean_s"] for s in ss]),
                "R5_latency_total_s": mean([s["R5_overhead"]["latency_total_s"] for s in ss]),
                "R5_tokens_est_total": mean([s["R5_overhead"]["tokens_est_total"] for s in ss]),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="all")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--backend-mode", choices=["openai", "hf_local"], required=True)
    ap.add_argument("--list-cells", action="store_true")
    args = ap.parse_args()

    pairs = parse_pairs(args.pairs)
    seeds = parse_seed_list(args.seeds)

    print("Resolved AgentRR cells:")
    for i, (d, m) in enumerate(pairs, 1):
        for seed in seeds:
            print(f"  {i}/{len(pairs)} {d}:{m}:seed{seed}")
    print("AgentRR backend mode:", args.backend_mode)

    if args.list_cells:
        return 0

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    summaries = []
    for d, m in pairs:
        for seed in seeds:
            print(f"Running AgentRR-style replay/check agent: {d}:{m}:seed{seed}")
            summaries.append(summarize_cell(d, m, seed, args.n_examples, args.backend_mode))

    manifest = {
        "baseline": "agentrr_style_record_replay_adapter",
        "backend_mode": args.backend_mode,
        "pairs": [{"dataset": d, "model": m} for d, m in pairs],
        "seeds": seeds,
        "n_examples": args.n_examples,
        "summary_files": [
            str(
                OUT_ROOT
                / f"{slug(s['dataset'])}__{slug(s['model'])}__seed{s['seed']}__n{s['n']}__{slug(s['backend_mode'])}"
                / "summary.json"
            )
            for s in summaries
        ],
    }

    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    agg = aggregate(summaries)
    (OUT_ROOT / "aggregate_by_dataset_model.json").write_text(json.dumps(agg, indent=2, sort_keys=True))

    out_csv = ROOT / "results/tables/csv/agentrr_outputs"
    out_csv.mkdir(parents=True, exist_ok=True)
    (out_csv / "official_agentrr_aggregates.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in agg),
        encoding="utf-8",
    )

    print("Completed AgentRR-style replay/check agent run.")
    print(OUT_ROOT / "manifest.json")
    print(OUT_ROOT / "aggregate_by_dataset_model.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
