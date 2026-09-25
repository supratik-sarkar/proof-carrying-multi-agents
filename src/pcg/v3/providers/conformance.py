"""Recorded-transport conformance suite.

Runs every hosted client end-to-end against frozen recorded responses with
NETWORK_API_MODEL_CALLS = 0. The pass criterion is behavioural, not existential.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .contract import (ProviderRequest, REDACTED, redact_headers)
from .hosted_clients import CLIENTS

FIXTURES = os.environ.get(
    "PCG_PROVIDER_FIXTURES",
    os.path.join("tests", "fixtures", "v3", "provider_transport"))

CHECKS = ("request_translation", "unsupported_params_dropped", "response_parsing",
          "model_identity", "usage_extraction", "tool_call_parsing", "error_mapping",
          "retry_classification", "raw_output_hash", "no_credentials_persisted",
          "fingerprint_stable")


def _load(name: str) -> Dict[str, Any]:
    with open(os.path.join(FIXTURES, f"{name}.json")) as fh:
        return json.load(fh)


def run_one(name: str) -> Dict[str, Any]:
    client = CLIENTS[name]
    fx = _load(name)
    req = ProviderRequest(model="requested-model-x", prompt="Does Policy X permit Y?",
                          system="be precise", seed=7, max_tokens=64,
                          tools=[{"name": "lookup"}])
    res: Dict[str, Any] = {}

    body = client.build_request(req)
    res["request_translation"] = isinstance(body, dict) and len(body) >= 3
    caps = client.capabilities()
    flat = json.dumps(body)
    res["unsupported_params_dropped"] = caps.supports_seed or ("seed" not in flat)

    r = client.parse_response(fx["ok"], req)
    res["response_parsing"] = bool(r.text) and "recorded" in r.text
    res["model_identity"] = (r.requested_model == "requested-model-x"
                             and r.returned_model not in (None, "")
                             and r.returned_model != r.requested_model)
    res["usage_extraction"] = (r.usage.input_tokens or 0) > 0 and (r.usage.output_tokens or 0) > 0
    expects_tools = any(k in json.dumps(fx["ok"]) for k in ("tool_calls", "tool_use", "functionCall"))
    res["tool_call_parsing"] = (len(r.tool_calls) > 0) if expects_tools else True

    err = client.map_error(fx["err"]["status"], fx["err"]["body"])
    res["error_mapping"] = err.error_class in (
        "AUTH", "RATE_LIMIT", "TIMEOUT", "TRANSPORT", "BAD_REQUEST", "UNKNOWN")
    st = fx["err"]["status"]
    should_retry = st == 429 or st in (408, 504) or 500 <= st < 600
    res["retry_classification"] = (err.retryable == should_retry) and (
        (err.retry_trigger_class in ("TRANSPORT", "RATE_LIMIT", "TIMEOUT"))
        if should_retry else err.retry_trigger_class is None)

    res["raw_output_hash"] = (len(r.raw_output_sha256) == 64
                              and r.raw_output_sha256 == client.parse_response(fx["ok"], req).raw_output_sha256)
    persisted = json.dumps(r.to_dict())
    hdrs = redact_headers({"Authorization": "Bearer sk-secret", "X-Trace": "ok"})
    res["no_credentials_persisted"] = ("sk-secret" not in persisted
                                       and hdrs["Authorization"] == REDACTED)
    res["fingerprint_stable"] = (r.backend_fingerprint ==
                                 client.parse_response(fx["ok"], req).backend_fingerprint)

    return {"provider": name, "checks": res, "passed": all(res.values()),
            "failed_checks": [k for k, v in res.items() if not v]}


def run_all() -> Dict[str, Any]:
    rows = [run_one(n) for n in CLIENTS]
    n_pass = sum(1 for r in rows if r["passed"])
    return {"providers": rows, "passing": n_pass, "total": len(rows),
            "network_api_model_calls": 0, "checks": list(CHECKS)}
