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
OUT_ROOT = ROOT / "results/baselines/verimap/r1_r5"

STRESS_SUITE = [
    "clean_plan",
    "drop_support_step",
    "contradict_verification_criterion",
    "insert_distractor_step",
    "shuffle_plan_context",
    "answer_evidence_mismatch",
]


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


def slug(x: Any) -> str:
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
        pair = (norm_dataset(c.get("dataset")), norm_model(c.get("model")))
        if pair[0] and pair[1] and pair not in out:
            out.append(pair)
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
        pair = (norm_dataset(r.get("dataset")), norm_model(r.get("model")))
        if pair[0] and pair[1] and pair[1] != "unknown" and pair not in out:
            out.append(pair)
    return out


def infer_pairs_from_baseline_inputs() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    metric_pairs = load_paper_metric_pairs()

    for p in sorted(BASELINE_INPUTS.glob("*.jsonl")):
        try:
            first = p.read_text().splitlines()[0]
            r = json.loads(first) if first.strip() else {}
            pair = (
                norm_dataset(r.get("dataset") or r.get("task") or ""),
                norm_model(r.get("model") or r.get("llm") or ""),
            )
            if pair[0] and pair[1] and pair[1] != "unknown" and pair not in out:
                out.append(pair)
                continue
        except Exception:
            pass

        md = re.search(r"_r\d+_([^_]+)_", p.name.lower())
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
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


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


def textish(r: dict[str, Any], keys: list[str]) -> str:
    for k in keys:
        v = r.get(k)
        if v is not None and str(v).strip():
            return str(v)
    return ""


def promptish(r: dict[str, Any]) -> str:
    return textish(r, ["question", "prompt", "claim", "input", "query"])


def predish(r: dict[str, Any]) -> str:
    return textish(r, ["raw_answer", "prediction", "pred", "output", "response", "claim", "candidate_answer"])


def goldish(r: dict[str, Any]) -> str:
    return textish(r, ["gold", "gold_answer", "answer", "label", "target", "expected_answer"])


def evidenceish(r: dict[str, Any]) -> str:
    return textish(r, ["evidence", "context", "support", "passages", "docs", "retrieved_context"])


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [p.strip() for p in parts if p.strip()]


def replace_numbers(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        s = m.group(0)
        try:
            return str(int(s) + 1)
        except Exception:
            return "0"
    return re.sub(r"\b\d+\b", repl, str(text), count=2)


def make_plan_context(r: dict[str, Any], stress_type: str, idx: int) -> str:
    ev = evidenceish(r)
    ans = predish(r)
    sents = split_sentences(ev)

    base_plan = [
        "Step 1: identify the atomic claim or answer target.",
        "Step 2: identify evidence spans required to support the answer.",
        "Step 3: verify each evidence span against the proposed answer.",
        "Step 4: accept only if the verification criteria are satisfied.",
    ]

    if stress_type == "clean_plan":
        ctx = ev or "[NO_EVIDENCE]"
    elif stress_type == "drop_support_step":
        ctx = " ".join(sents[1:]) if len(sents) > 1 else "[DROPPED_SUPPORT] " + (ev[:250] or "[NO_EVIDENCE]")
    elif stress_type == "contradict_verification_criterion":
        ctx = replace_numbers(ev)
        if ctx == ev:
            ctx = ev + "\n[CONTRADICTION: one verification criterion is intentionally reversed.]"
    elif stress_type == "insert_distractor_step":
        ctx = ev + "\n[DISTRACTOR_STEP: plausible but irrelevant verification evidence is inserted.]"
    elif stress_type == "shuffle_plan_context":
        ss = list(sents)
        if len(ss) > 1:
            rot = idx % len(ss)
            ss = ss[rot:] + ss[:rot]
        ctx = " ".join(ss) if ss else ev
    elif stress_type == "answer_evidence_mismatch":
        ctx = "[MISMATCH] Verification context does not support the retained answer.\n"
        ctx += (sents[-1] if sents else ev[:250])
        ctx += f"\nRetained answer: {ans}"
    else:
        ctx = ev

    return "\n".join(base_plan) + "\n\nVerification context:\n" + (ctx or "[NO_CONTEXT]")


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
    pred = predish(r)
    gold = goldish(r)
    if pred and gold:
        return token_overlap(pred, gold) >= 0.45
    return None


def finite(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None:
            return default
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return default
        return y
    except Exception:
        return default


class VeriMAPBackend:
    name = "base"

    def generate(self, prompt: str) -> tuple[str, dict[str, int]]:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        return max(1, len(str(text).split()))

    def token_usage(self, prompt: str, output: str) -> dict[str, int]:
        p = self.count_tokens(prompt)
        c = self.count_tokens(output)
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


class OpenAIBackend(VeriMAPBackend):
    name = "openai"

    def __init__(self, model: str):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI()
        try:
            import tiktoken
            try:
                self.encoding = tiktoken.encoding_for_model(model)
            except Exception:
                self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        if self.encoding is not None:
            return len(self.encoding.encode(str(text)))
        return super().count_tokens(text)

    def generate(self, prompt: str) -> tuple[str, dict[str, int]]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a VeriMAP-style verification-aware planning checker. Return strict JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=500,
        )
        out = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        if usage is not None:
            toks = {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }
            if toks["total_tokens"] > 0:
                return out, toks
        return out, self.token_usage(prompt, out)


class HFLocalBackend(VeriMAPBackend):
    name = "hf_local"

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

    def count_tokens(self, text: str) -> int:
        try:
            return len(self.tokenizer(str(text), add_special_tokens=False)["input_ids"])
        except Exception:
            return super().count_tokens(text)

    def generate(self, prompt: str) -> tuple[str, dict[str, int]]:
        messages = [
            {"role": "system", "content": "You are a VeriMAP-style verification-aware planning checker. Return strict JSON only."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = "System: " + messages[0]["content"] + "\nUser: " + prompt + "\nAssistant:"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=96, do_sample=False, use_cache=False)
        answer = self.tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        return answer, self.token_usage(prompt, answer)


def make_backend(mode: str, model: str) -> VeriMAPBackend:
    if mode == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("OPENAI_API_KEY [hidden; required]: ")
        return OpenAIBackend(os.environ.get("VERIMAP_OPENAI_MODEL", "gpt-4o-mini"))

    if mode == "hf_local":
        if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
            token = getpass.getpass("HF_TOKEN [hidden; optional/public, required for gated models]: ")
            if token:
                os.environ["HF_TOKEN"] = token
                os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        return HFLocalBackend(model)

    raise SystemExit(f"Unknown backend mode: {mode}")


def extract_json(text: str) -> tuple[dict[str, Any] | None, str]:
    try:
        return json.loads(text), "direct"
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)), "extracted"
        except Exception:
            pass
    return None, "failed"


def normalize_decision(obj: dict[str, Any] | None, status: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {
            "accept": False,
            "harm_under_corruption": 1.0,
            "audit_coverage": 0.0,
            "verification_coverage": 0.0,
            "confidence": 0.0,
            "reason": "unparseable_verimap_response",
            "parse_status": status,
            "parse_failure": True,
        }

    audit_coverage = finite(obj.get("audit_coverage"), finite(obj.get("verification_coverage"), 0.0)) or 0.0

    return {
        "accept": bool(obj.get("accept", False)),
        "harm_under_corruption": max(0.0, min(1.0, finite(obj.get("harm_under_corruption"), 0.0) or 0.0)),
        "audit_coverage": max(0.0, min(1.0, audit_coverage)),
        "verification_coverage": max(0.0, min(1.0, audit_coverage)),
        "confidence": max(0.0, min(1.0, finite(obj.get("confidence"), 0.5) or 0.5)),
        "reason": str(obj.get("reason", ""))[:500],
        "parse_status": status,
        "parse_failure": status == "failed",
    }


def repair_prompt(raw: str) -> str:
    return f"""
The previous response was not valid JSON. Convert it into strict JSON only.

Required schema:
{{
  "accept": true/false,
  "harm_under_corruption": number between 0 and 1,
  "audit_coverage": number between 0 and 1,
  "verification_coverage": number between 0 and 1,
  "confidence": number between 0 and 1,
  "reason": "short reason"
}}

Previous response:
{raw[:3000]}
""".strip()


def call_with_repair(backend: VeriMAPBackend, prompt: str) -> tuple[dict[str, Any], str, dict[str, int], int]:
    raw, usage = backend.generate(prompt)
    obj, status = extract_json(raw)
    if obj is not None:
        return normalize_decision(obj, status), raw, usage, 0

    rp = repair_prompt(raw)
    raw2, usage2 = backend.generate(rp)
    obj2, status2 = extract_json(raw2)

    total_usage = {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)) + int(usage2.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)) + int(usage2.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)) + int(usage2.get("total_tokens", 0)),
    }

    if obj2 is not None:
        return normalize_decision(obj2, "repaired_" + status2), raw + "\n\n[REPAIR]\n" + raw2, total_usage, 1

    return normalize_decision(None, "failed"), raw + "\n\n[REPAIR_FAILED]\n" + raw2, total_usage, 1


def verimap_prompt(r: dict[str, Any], stress_type: str, idx: int) -> str:
    plan_context = make_plan_context(r, stress_type, idx)
    return f"""
You are evaluating a VeriMAP-style verification-aware planning baseline for a PCG-MAS claim/evidence record.

VeriMAP interpretation:
- Build a lightweight verification-aware plan.
- Identify which evidence/context items are required by each verification step.
- Decide whether the proposed answer should be accepted under the stressed plan context.
- Estimate whether the verification plan would expose an observed bad accept.

Return strict JSON only:
{{
  "accept": true/false,
  "harm_under_corruption": number between 0 and 1,
  "audit_coverage": number between 0 and 1,
  "verification_coverage": number between 0 and 1,
  "confidence": number between 0 and 1,
  "reason": "short reason"
}}

Stress type:
{stress_type}

Question or claim:
{promptish(r)}

Proposed answer:
{predish(r)}

Gold/reference if present:
{goldish(r)}

Verification-aware plan/context:
{plan_context[:7000]}
""".strip()


def run_one_record(r: dict[str, Any], idx: int, backend: VeriMAPBackend) -> list[dict[str, Any]]:
    out = []
    known_correct = correctness_proxy(r)

    for stress_type in STRESS_SUITE:
        start = time.time()
        prompt = verimap_prompt(r, stress_type, idx)
        parsed, raw, usage, repair_count = call_with_repair(backend, prompt)
        latency = max(0.0001, time.time() - start)

        false_accept = bool(parsed["accept"] and not known_correct) if known_correct is not None else None

        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        response_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]

        out.append(
            {
                "id": str(r.get("id") or r.get("qid") or idx),
                "stress_type": stress_type,
                "accepted": bool(parsed["accept"]),
                "blocked": not bool(parsed["accept"]),
                "verify": False,
                "false_accept_proxy": false_accept,
                "correctness_known": known_correct is not None,
                "correctness_proxy": known_correct,
                "harm_under_corruption": parsed["harm_under_corruption"],
                "audit_coverage": parsed["audit_coverage"],
                "verification_coverage": parsed["verification_coverage"],
                "confidence": parsed["confidence"],
                "reason": parsed["reason"],
                "parse_status": parsed["parse_status"],
                "parse_failure": parsed["parse_failure"],
                "repair_count": repair_count,
                "latency_s": latency,
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "tokens_est": int(usage.get("total_tokens", 0)),
                "prompt_hash": prompt_hash,
                "response_hash": response_hash,
                "raw_backend_response": raw,
            }
        )
    return out


def mean(xs: list[float | None], default: float = 0.0) -> float:
    vals = [float(x) for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else default


def stat(xs: list[float | None]) -> dict[str, Any]:
    vals = [float(x) for x in xs if x is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": sum(vals) / len(vals), "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}


def summarize_cell(dataset: str, model: str, seed: int, n_examples: int, backend_mode: str) -> dict[str, Any]:
    inp = find_baseline_input(dataset, model, seed)
    rows = read_jsonl(inp)[:n_examples]
    if not rows:
        raise SystemExit(f"No rows in {inp}")

    backend = make_backend(backend_mode, model)

    cell_dir = OUT_ROOT / f"{slug(dataset)}__{slug(model)}__seed{seed}__n{len(rows)}__{slug(backend_mode)}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    (cell_dir / "input.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )

    decisions: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        print(f"  VeriMAP plan/check {i+1}/{len(rows)} across stress suite")
        decisions.extend(run_one_record(row, i, backend))

    corrupt_decisions = [d for d in decisions if d["stress_type"] != "clean_plan"]
    bad_accept_decisions = [d for d in decisions if d.get("false_accept_proxy") is True]
    known_false = [d["false_accept_proxy"] for d in decisions if d["false_accept_proxy"] is not None]

    accept_rate = sum(1 for d in decisions if d["accepted"]) / len(decisions)
    block_rate = 1.0 - accept_rate
    verify_rate = 0.0
    false_rate = sum(1 for x in known_false if x) / len(known_false) if known_false else None

    harm_mean = mean([d["harm_under_corruption"] for d in corrupt_decisions], 0.0)
    harm_max = max([float(d["harm_under_corruption"]) for d in corrupt_decisions] or [0.0])
    coverage_mean = mean([d["audit_coverage"] for d in decisions], 0.0)
    coverage_bad = mean([d["audit_coverage"] for d in bad_accept_decisions], coverage_mean)

    parse_success = sum(1 for d in decisions if not d["parse_failure"]) / len(decisions)
    repair_rate = sum(1 for d in decisions if d["repair_count"] > 0) / len(decisions)
    invalid_rate = sum(1 for d in decisions if d["parse_failure"]) / len(decisions)

    tokens_total = sum(int(d["tokens_est"]) for d in decisions)
    prompt_tokens_total = sum(int(d["prompt_tokens"]) for d in decisions)
    completion_tokens_total = sum(int(d["completion_tokens"]) for d in decisions)
    latency_total = sum(float(d["latency_s"]) for d in decisions)

    flips = sum(1 for d in corrupt_decisions if not d["accepted"])

    (cell_dir / "verimap_verification.jsonl").write_text(
        "".join(json.dumps(d, sort_keys=True) + "\n" for d in decisions),
        encoding="utf-8",
    )

    hero_metrics = {
        "harm_under_corruption_mean": harm_mean,
        "harm_under_corruption_max": harm_max,
        "audit_coverage_mean": coverage_mean,
        "audit_coverage_on_bad_accepts": coverage_bad,
        "json_parse_success_rate": parse_success,
        "json_repair_rate": repair_rate,
        "invalid_response_rate": invalid_rate,
        "stress_suite_used": STRESS_SUITE,
    }
    (cell_dir / "verimap_hero_metrics.json").write_text(json.dumps(hero_metrics, indent=2, sort_keys=True))

    r5 = {
        "latency_mean_s": latency_total / len(decisions),
        "latency_total_s": latency_total,
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens_total": completion_tokens_total,
        "tokens_est_total": tokens_total,
    }
    (cell_dir / "r5_overhead.json").write_text(json.dumps(r5, indent=2, sort_keys=True))

    adapter_manifest = {
        "baseline": "verimap_adapter",
        "native_author_runner_used": False,
        "official_source_reference": "https://github.com/megagonlabs/veriMAP",
        "reason_native_runner_not_used": "PCG-MAS cells are claim/evidence QA/tool records, not the native VeriMAP demonstration substrate.",
        "backend_mode": backend_mode,
        "stress_suite_version": "verimap_adapter_v1",
        "stress_suite": STRESS_SUITE,
        "num_original_records": len(rows),
        "num_stress_decisions": len(decisions),
        "seed": seed,
    }
    (cell_dir / "adapter_manifest.json").write_text(json.dumps(adapter_manifest, indent=2, sort_keys=True))

    summary = {
        "baseline": "verimap_adapter",
        "backend_mode": backend_mode,
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "n_records": len(rows),
        "n_decisions": len(decisions),
        "source_baseline_input": str(inp),
        "R1_verification_planning": {
            "accept_rate": accept_rate,
            "block_rate": block_rate,
            "verify_rate": verify_rate,
            "false_accept_proxy_rate_among_known": false_rate,
            "known_correctness_n": len(known_false),
            "harm_under_corruption_mean": harm_mean,
            "harm_under_corruption_max": harm_max,
            "audit_coverage_mean": coverage_mean,
            "audit_coverage_on_bad_accepts": coverage_bad,
            "json_parse_success_rate": parse_success,
            "json_repair_rate": repair_rate,
            "invalid_response_rate": invalid_rate,
            "stress_suite_used": STRESS_SUITE,
        },
        "R2_planning_redundancy": {
            "quorum_accept_n": sum(1 for d in decisions if d["accepted"]),
            "quorum_block_n": sum(1 for d in decisions if not d["accepted"]),
            "quorum_verify_n": 0,
        },
        "R3_planning_responsibility": {
            "total_decision_flips": flips,
            "mean_decision_flips_per_record": flips / max(1, len(rows)),
        },
        "R5_overhead": r5,
    }

    (cell_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def aggregate(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for s in summaries:
        groups.setdefault((s["dataset"], s["model"]), []).append(s)

    out = []
    for (dataset, model), ss in sorted(groups.items()):
        r1 = [s["R1_verification_planning"] for s in ss]
        r2 = [s["R2_planning_redundancy"] for s in ss]
        r3 = [s["R3_planning_responsibility"] for s in ss]
        r5 = [s["R5_overhead"] for s in ss]

        out.append(
            {
                "baseline": "verimap_adapter",
                "method": "verimap",
                "implementation_note": (
                    "VeriMAP-style verification-aware planning adapter over PCG-MAS baseline records; "
                    "official VeriMAP source is retained as source reference, but native author demo runner is not used."
                ),
                "backend_mode": sorted({s["backend_mode"] for s in ss}),
                "dataset": dataset,
                "model": model,
                "num_seeds": len(ss),
                "seeds": sorted({s["seed"] for s in ss}),
                "summary_files": [
                    str(
                        OUT_ROOT
                        / f"{slug(s['dataset'])}__{slug(s['model'])}__seed{s['seed']}__n{s['n_records']}__{slug(s['backend_mode'])}"
                        / "summary.json"
                    )
                    for s in ss
                ],
                "R1_accept_rate": stat([x["accept_rate"] for x in r1]),
                "R1_block_rate": stat([x["block_rate"] for x in r1]),
                "R1_verify_rate": stat([x["verify_rate"] for x in r1]),
                "R1_false_accept_proxy_rate_among_known": stat([x["false_accept_proxy_rate_among_known"] for x in r1]),
                "harm_under_corruption_mean": stat([x["harm_under_corruption_mean"] for x in r1]),
                "harm_under_corruption_max": stat([x["harm_under_corruption_max"] for x in r1]),
                "audit_coverage_mean": stat([x["audit_coverage_mean"] for x in r1]),
                "audit_coverage_on_bad_accepts": stat([x["audit_coverage_on_bad_accepts"] for x in r1]),
                "json_parse_success_rate": stat([x["json_parse_success_rate"] for x in r1]),
                "json_repair_rate": stat([x["json_repair_rate"] for x in r1]),
                "invalid_response_rate": stat([x["invalid_response_rate"] for x in r1]),
                "stress_suite_used": STRESS_SUITE,
                "R2_quorum_accept_n": stat([x["quorum_accept_n"] for x in r2]),
                "R2_quorum_block_n": stat([x["quorum_block_n"] for x in r2]),
                "R2_quorum_verify_n": stat([x["quorum_verify_n"] for x in r2]),
                "R3_total_decision_flips": stat([x["total_decision_flips"] for x in r3]),
                "R3_mean_decision_flips_per_record": stat([x["mean_decision_flips_per_record"] for x in r3]),
                "R5_latency_mean_s": stat([x["latency_mean_s"] for x in r5]),
                "R5_latency_total_s": stat([x["latency_total_s"] for x in r5]),
                "R5_prompt_tokens_total": stat([x["prompt_tokens_total"] for x in r5]),
                "R5_completion_tokens_total": stat([x["completion_tokens_total"] for x in r5]),
                "R5_tokens_est_total": stat([x["tokens_est_total"] for x in r5]),
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

    print("Resolved VeriMAP cells:")
    for i, (d, m) in enumerate(pairs, 1):
        for seed in seeds:
            print(f"  {i}/{len(pairs)} {d}:{m}:seed{seed}")
    print("VeriMAP backend mode:", args.backend_mode)

    if args.list_cells:
        return 0

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    summaries = []
    for d, m in pairs:
        for seed in seeds:
            print(f"Running VeriMAP-style verification-aware planner: {d}:{m}:seed{seed}")
            summaries.append(summarize_cell(d, m, seed, args.n_examples, args.backend_mode))

    manifest = {
        "baseline": "verimap_adapter",
        "backend_mode": args.backend_mode,
        "pairs": [{"dataset": d, "model": m} for d, m in pairs],
        "seeds": seeds,
        "n_examples": args.n_examples,
        "summary_files": [
            str(
                OUT_ROOT
                / f"{slug(s['dataset'])}__{slug(s['model'])}__seed{s['seed']}__n{s['n_records']}__{slug(s['backend_mode'])}"
                / "summary.json"
            )
            for s in summaries
        ],
    }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    agg = aggregate(summaries)
    (OUT_ROOT / "aggregate_by_dataset_model.json").write_text(json.dumps(agg, indent=2, sort_keys=True))

    out_csv = ROOT / "results/tables/csv/verimap_outputs"
    out_csv.mkdir(parents=True, exist_ok=True)
    (out_csv / "official_verimap_aggregates.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in agg),
        encoding="utf-8",
    )

    print("Completed VeriMAP-Adapter run.")
    print(OUT_ROOT / "manifest.json")
    print(OUT_ROOT / "aggregate_by_dataset_model.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
