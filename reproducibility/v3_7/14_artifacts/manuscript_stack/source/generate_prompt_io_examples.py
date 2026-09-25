#!/usr/bin/env python3

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "reproducibility" / "v3_6"
OUT = ROOT / "manuscript" / "tables" / "generated_prompt_io_examples.tex"

INPUT_MANIFEST = REPRO / "03_prompts" / "generation_input_manifest.jsonl"
IO_BINDINGS = REPRO / "03_prompts" / "generation_io_bindings.jsonl"
GEN_MANIFEST = REPRO / "05_generations" / "generation_manifest.jsonl"
CERT_LEDGER = REPRO / "08_certificates" / "certificate_ledger.jsonl"
DECISIONS = REPRO / "08_certificates" / "acceptance_decisions.jsonl"
HARM_LABELS = REPRO / "09_outcomes" / "harm_labels.csv"

BUCKET2 = ROOT / "Bucket_2"
RAW_MANIFEST = REPRO / "05_generations" / "raw_response_manifest.jsonl"
VERIFIER_INPUT = REPRO / "07_verification" / "verifier_input_manifest.jsonl"
TOOL_CALL_MANIFEST = REPRO / "06_agent_traces" / "tool_call_manifest.jsonl"
TRAJECTORY_MANIFEST = REPRO / "06_agent_traces" / "trajectory_manifest.jsonl"


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def load_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def first(d, *names, default=None):
    for name in names:
        if isinstance(d, dict) and name in d:
            val = d[name]
            if val is not None and val != "":
                return val
    return default


def key(row):
    return (
        str(first(row, "model_id", "model", default="")),
        str(first(row, "dataset_id", "dataset", default="")),
        str(first(row, "observation_id", "observation", "task_id", default="")),
    )


def latex_escape(value, max_chars=None):
    if value is None:
        return r"\textsc{N/A}"

    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)

    s = str(value).strip()

    if max_chars and len(s) > max_chars:
        s = s[: max_chars - 3].rstrip() + "..."

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    escaped = "".join(replacements.get(ch, ch) for ch in s)
    return add_soft_breaks(escaped)


def add_soft_breaks(s: str) -> str:
    """Add zero-width LaTeX break opportunities without changing visible text."""
    for token in [",", ":", "/", "-", ";"]:
        s = s.replace(token, token + r"\allowbreak{}")

    # latex_escape converts underscore to \_
    s = s.replace(r"\_", r"\_\allowbreak{}")
    return s


def breakable_sha(value) -> str:
    """Render full SHA while allowing a line break every 8 hex characters."""
    if not value or str(value) in {"NOT_CAPTURED", "NOT_APPLICABLE", "None", "--"}:
        return "--"

    s = str(value).strip()
    chunks = [s[i:i+8] for i in range(0, len(s), 8)]
    return r"\allowbreak{}".join(chunks)


def clean_sha(value):
    if not value or str(value) in {
        "NOT_CAPTURED",
        "NOT_APPLICABLE",
        "None",
        "--",
    }:
        return "--"

    return breakable_sha(value)


def state_tex(channel_name, value):
    if isinstance(value, dict):
        value = first(
            value,
            "state",
            "value",
            "effective_contribution",
            "pass",
            default="N/A",
        )

    if value is True or str(value).upper() in {"1", "PASS", "TRUE", "ACCEPT"}:
        st = r"\textsc{Pass}"
    elif value is False or str(value).upper() in {"0", "FAIL", "FALSE", "REJECT"}:
        st = r"\textsc{Fail}"
    else:
        st = r"\textsc{N/A}"

    return rf"{channel_name}={st}"


inputs = load_jsonl(INPUT_MANIFEST)
bindings = load_jsonl(IO_BINDINGS)
generations = load_jsonl(GEN_MANIFEST)
certs = load_jsonl(CERT_LEDGER)
decisions = load_jsonl(DECISIONS)
outcomes = load_csv(HARM_LABELS)

input_by_id = {
    str(first(r, "input_id", "prompt_id", default="")): r for r in inputs
}

gen_by_key = {key(r): r for r in generations}
cert_by_key = {key(r): r for r in certs}
decision_by_key = {key(r): r for r in decisions}
outcome_by_key = {key(r): r for r in outcomes}

binding_by_key = {key(r): r for r in bindings}

# Overlay candidate_text from verifier_input_manifest if present
if VERIFIER_INPUT.exists():
    for r in load_jsonl(VERIFIER_INPUT):
        k = key(r)
        if k in gen_by_key and r.get("candidate_text"):
            gen_by_key[k]["candidate_text"] = r["candidate_text"]

# Overlay raw outputs (candidate_text, text, tool_calls) from raw_response_manifest
if RAW_MANIFEST.exists():
    for r in load_jsonl(RAW_MANIFEST):
        k = key(r)
        rel = r.get("raw_response_rel_path")
        if not rel:
            continue
        for base in (BUCKET2, REPRO, ROOT):
            p = base / rel
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    g = gen_by_key.setdefault(k, {})
                    for field in ("candidate_text", "text", "tool_calls"):
                        if field in data and data[field]:
                            g[field] = data[field]
                except Exception:
                    pass
                break

# Overlay agent traces from tool_call_manifest / trajectory_manifest for agentdojo
traj_to_key = {}
if TRAJECTORY_MANIFEST.exists():
    for r in load_jsonl(TRAJECTORY_MANIFEST):
        u = r.get("user_task_id")
        i = r.get("injection_task_id")
        s = r.get("suite_name")
        obs = f"agentdojo:{s}:{u}:{i}" if i else f"agentdojo:{s}:{u}"
        traj_to_key[r.get("trajectory_id")] = (str(r.get("model_id")), "agentdojo", obs)

if TOOL_CALL_MANIFEST.exists():
    calls_by_traj = {}
    for r in load_jsonl(TOOL_CALL_MANIFEST):
        tid = r.get("trajectory_id")
        calls_by_traj.setdefault(tid, []).append({
            "function": r.get("function_name"),
            "args": r.get("args"),
        })
    for tid, calls in calls_by_traj.items():
        if tid in traj_to_key:
            k = traj_to_key[tid]
            if k in gen_by_key and not gen_by_key[k].get("tool_calls"):
                gen_by_key[k]["tool_calls"] = calls



def choose_example(dataset, require_published_text=False):
    candidates = []

    for b in bindings:
        if str(first(b, "dataset_id", "dataset", default="")) != dataset:
            continue

        k = key(b)
        g = gen_by_key.get(k, {})
        c = cert_by_key.get(k, {})
        d = decision_by_key.get(k, {})
        o = outcome_by_key.get(k, {})

        input_id = str(first(b, "input_id", default=""))
        inp = input_by_id.get(input_id, {})

        provenance = str(first(inp, "provenance_status", default=""))

        if require_published_text and provenance != "CAPTURED":
            continue

        status = str(first(g, "status", "generation_status", default="")).upper()
        if status not in {"SUCCESS", "COMPLETE", "COMPLETED", "OK", ""}:
            continue

        candidate = first(
            g,
            "candidate_text",
            "text",
            "output_text",
            "response",
            "candidate",
            default=None,
        )

        structured = first(
            g,
            "structured_output",
            "tool_call",
            "tool_calls",
            "action",
            "trajectory",
            "output_payload",
            default=None,
        )

        if candidate in (None, "") and structured in (None, "", [], {}):
            continue

        candidates.append((k, b, inp, g, c, d, o))

    if not candidates:
        raise RuntimeError(f"No usable example found for dataset={dataset}")

    # Deterministic choice: lexicographically smallest canonical key.
    candidates.sort(key=lambda x: x[0])
    return candidates[0]


def input_text(inp):
    # Prefer actual captured text fields if present.
    value = first(
        inp,
        "user_or_task_input",
        "user_prompt",
        "prompt_text",
        "text",
        "input_text",
        "task_text",
        "semantic_input",
        default=None,
    )

    if value:
        return latex_escape(value, max_chars=650)

    path = first(
        inp,
        "user_or_task_input_path",
        "system_prompt_path",
        default=None,
    )

    if path:
        p = REPRO / str(path)
        if p.exists() and p.is_file():
            return latex_escape(p.read_text(encoding="utf-8"), max_chars=650)

    return "Exact input text unavailable in this rendered view."


def generated_output(g):
    value = first(
        g,
        "candidate_text",
        "text",
        "output_text",
        "response",
        "candidate",
        "structured_output",
        "tool_call",
        "tool_calls",
        "action",
        "trajectory",
        "output_payload",
        default="--",
    )
    return latex_escape(value, max_chars=650)


def cert_channels(c):
    # Support several plausible historical schemas.
    factors = first(c, "factors", "channels", "checks", default={})
    if not isinstance(factors, dict):
        factors = {}

    vh = first(c, "V_H", "v_h", default=first(factors, "V_H", "v_h"))
    vpi = first(c, "V_Pi", "V_PI", "v_pi", default=first(factors, "V_Pi", "V_PI", "v_pi"))
    vgamma = first(
        c,
        "V_Gamma",
        "V_GAMMA",
        "v_gamma",
        default=first(factors, "V_Gamma", "V_GAMMA", "v_gamma"),
    )
    vent = first(
        c,
        "V_vdash",
        "V_entail",
        "v_entail",
        default=first(factors, "V_vdash", "V_entail", "v_entail"),
    )

    return (
        state_tex(r"V_H", vh),
        state_tex(r"V_\Pi", vpi),
        state_tex(r"V_\Gamma", vgamma),
        state_tex(r"V_\vdash", vent),
    )


def decision_text(d, c):
    value = first(
        d,
        "pcg_accepted",
        "decision",
        "acceptance_decision",
        "accepted",
        "accept",
        default=first(c, "pcg_accepted", "decision", "accepted", "accept", default="--"),
    )

    if value is True or str(value).upper() in {"1", "TRUE", "ACCEPT", "ACCEPTED"}:
        return r"\textsc{Accept}"
    if value is False or str(value).upper() in {"0", "FALSE", "REJECT", "REJECTED"}:
        return r"\textsc{Reject}"
    return latex_escape(value)


def outcome_text(o):
    if not o:
        return "--"

    value = first(
        o,
        "outcome",
        "label",
        "harm_label",
        "native_outcome",
        "benchmark_outcome",
        "utility_score",
        "harmful",
        default=None,
    )

    if value is None:
        # Compact representation if schema differs.
        filtered = {
            k: v
            for k, v in o.items()
            if k not in {"model_id", "dataset_id", "observation_id"}
            and v not in {"", None}
        }
        return latex_escape(filtered, max_chars=250)

    return latex_escape(value, max_chars=250)


def output_sha(b, g):
    if first(b, "output_payload_semantics") == "STRUCTURED_ACTION_TRAJECTORY":
        s_sha = first(
            b,
            "structured_output_sha256",
            default=first(g, "structured_output_sha256"),
        )
        if s_sha and s_sha not in {"None", "NOT_CAPTURED", "NOT_APPLICABLE"}:
            return clean_sha(s_sha)

    c_sha = first(
        b,
        "candidate_output_sha256",
        "candidate_sha256",
        "output_sha256",
        default=first(
            g,
            "candidate_sha256",
            "content_sha256",
            "output_sha256",
        ),
    )
    if c_sha == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855":
        s_sha = first(
            b,
            "structured_output_sha256",
            default=first(g, "structured_output_sha256"),
        )
        if s_sha and s_sha not in {"None", "NOT_CAPTURED", "NOT_APPLICABLE"}:
            return clean_sha(s_sha)

    return clean_sha(
        c_sha
        or first(
            b,
            "structured_output_sha256",
            default=first(g, "structured_output_sha256", default="--"),
        )
    )


def semantic_input_sha(b, inp):
    return clean_sha(
        first(
            b,
            "semantic_input_sha256",
            "prompt_sha256",
            default=first(
                inp,
                "semantic_input_sha256",
                "user_or_task_input_sha256",
                "prompt_sha256",
                default="--",
            ),
        )
    )


# Pick one stable example from each interface.
fact = choose_example("hotpotqa", require_published_text=True)
dojo = choose_example("agentdojo", require_published_text=True)
bfcl = choose_example("bfcl_v1", require_published_text=False)
m2w = choose_example("mind2web", require_published_text=False)


def render_group(prefix, item, include_input_text=True):
    k, b, inp, g, c, d, o = item
    model_id, dataset_id, observation_id = k
    vh, vpi, vgamma, vent = cert_channels(c)

    lines = []

    def cmd(name, value):
        lines.append(rf"\newcommand{{\{prefix}{name}}}{{{value}}}")

    cmd("Identity", latex_escape(f"{dataset_id} / {observation_id}"))
    cmd("Model", latex_escape(model_id))

    if include_input_text:
        cmd("Input", input_text(inp))
    else:
        cmd("InputSHA", semantic_input_sha(b, inp))

    cmd("Output", generated_output(g))
    cmd("OutputSHA", output_sha(b, g))
    cmd("VH", vh)
    cmd("VPi", vpi)
    cmd("VGamma", vgamma)
    cmd("VEntail", vent)
    cmd("Decision", decision_text(d, c))
    cmd("Outcome", outcome_text(o))

    return lines


tex = [
    "% AUTO-GENERATED FILE.",
    "% Do not hand-edit scientific values.",
    "% Source: reproducibility/v3_6 canonical prompt/I-O/certificate/outcome records.",
    "",
]

tex += render_group("PCGExampleFact", fact, include_input_text=True)
tex += [""]
tex += render_group("PCGExampleDojo", dojo, include_input_text=True)
tex += [""]
tex += render_group("PCGExampleBFCL", bfcl, include_input_text=False)
tex += [""]
tex += render_group("PCGExampleMindTwoWeb", m2w, include_input_text=False)
tex += [""]

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(tex), encoding="utf-8")

required_prefixes = [
    "PCGExampleFact",
    "PCGExampleDojo",
    "PCGExampleBFCL",
    "PCGExampleMindTwoWeb",
]

content = OUT.read_text(encoding="utf-8")

for prefix in required_prefixes:
    if f"\\newcommand{{\\{prefix}" not in content:
        raise RuntimeError(f"Generation failed for {prefix}")

print(f"WROTE={OUT}")
print("PROMPT_IO_EXAMPLE_MACROS=PASS")
