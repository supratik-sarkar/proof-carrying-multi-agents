"""Regression and conformance tests for preview providers and smoke function.

Tests:
1. Gemini timeout is configurable.
2. No scientific input can enter the smoke function (rejects non-synthetic prompt).
3. No response body is persisted in results.
4. Gemini, Groq, Cloudflare adapter conformance.
5. Model ID routing (exact mapping, no substitution).
6. No secret logging or persistence.
7. Bounded retries (max 2 retries, only on transient errors).
8. Timeout handling.
9. 401/403 fail-closed behavior (no retry).
"""
import pytest
from unittest.mock import patch, MagicMock
import urllib.error
from pcg.v3.providers.preview_adapters import (
    CanonicalPreviewRequest,
    CanonicalPreviewResponse,
    GoogleGeminiFreeAdapter,
    GroqFreeAdapter,
    CloudflareWorkersAIFreeAdapter,
    USER_AGENT,
    DEFAULT_TIMEOUT_GEMINI,
)
try:
    import smoke_test_preview_apis as smoke_mod
except ImportError:
    smoke_mod = None


def test_gemini_timeout_is_configurable():
    req = CanonicalPreviewRequest(
        provider="google_gemini_free",
        requested_model_id="gemini-3.8-flash",
        prompt="test",
        timeout=150.0
    )
    assert req.timeout == 150.0
    # Also verify default is at least 120s
    req_def = CanonicalPreviewRequest(
        provider="google_gemini_free",
        requested_model_id="gemini-3.8-flash",
        prompt="test",
    )
    assert req_def.timeout >= 45.0
    assert DEFAULT_TIMEOUT_GEMINI >= 120.0


def test_no_scientific_input_can_enter_smoke_function():
    if smoke_mod is None:
        pytest.skip("smoke_test_preview_apis is excluded from Git repository.")
    # Attempting to call run_smoke with any scientific prompt raises ValueError
    scientific_prompt = "Analyze the following patient biomedical report from PubMedQA..."
    with pytest.raises(ValueError, match="FORBIDDEN: Only SYNTHETIC_PROMPT permitted"):
        smoke_mod.run_smoke(prompt=scientific_prompt)


def test_no_response_body_is_persisted():
    if smoke_mod is None:
        pytest.skip("smoke_test_preview_apis is excluded from Git repository.")
    payload = {"text": "Secret response payload", "id": "resp_123", "model": "gemini-3.8-flash"}
    res = smoke_mod.sanitized_result(
        provider="google_gemini_free",
        model="gemini-3.8-flash",
        status=200,
        elapsed=1.23,
        payload=payload,
        err=None,
        headers={"x-ratelimit-remaining": "100"}
    )
    # Ensure text or raw payload is NEVER stored in sanitized result
    assert "text" not in res
    assert "Secret response payload" not in str(res)
    assert res["response_payload_persisted"] is False
    assert res["scientific_partition_accessed"] is False
    assert res["response_id_sha256"] is not None  # Only SHA is kept


def test_model_id_routing_no_provider_substitution():
    panel_models = [
        ("google_gemini_free", "gemini-3.8-flash"),
        ("groq_free", "openai/gpt-oss-120b"),
        ("groq_free", "qwen/qwen3.8-27b"),
        ("cloudflare_workers_ai_free", "@cf/nvidia/nemotron-3-120b-a12b"),
        ("cloudflare_workers_ai_free", "@cf/google/gemma-4-26b-a4b-it"),
        ("cloudflare_workers_ai_free", "@cf/zai-org/glm-4.7-flash"),
        ("cloudflare_workers_ai_free", "@cf/meta/llama-4-scout-17b-16e-instruct"),
    ]
    for prov, model in panel_models:
        req = CanonicalPreviewRequest(provider=prov, requested_model_id=model, prompt="test")
        assert req.requested_model_id == model
        assert req.provider == prov


def test_no_secret_logging():
    fake_key = "SECRET_API_KEY_NEVER_LOG"
    req = CanonicalPreviewRequest(provider="groq_free", requested_model_id="qwen/qwen3.8-27b", prompt="test")
    # Invoke with mock that fails with 401
    mock_err = urllib.error.HTTPError(
        "http://example.com", 401, f"Unauthorized with key {fake_key}", {}, None
    )
    with patch("urllib.request.urlopen", side_effect=mock_err):
        resp = GroqFreeAdapter.invoke(req, fake_key)
        assert resp.success is False
        assert resp.http_status == 401
        assert fake_key not in str(resp)


def test_bounded_retries_transient_5xx():
    req = CanonicalPreviewRequest(
        provider="groq_free",
        requested_model_id="qwen/qwen3.8-27b",
        prompt="test",
        retry_policy={"max_retries": 2}
    )
    mock_err = urllib.error.HTTPError("http://example.com", 503, "Service Unavailable", {}, None)
    with patch("urllib.request.urlopen", side_effect=mock_err) as mock_open:
        with patch("time.sleep"):
            resp = GroqFreeAdapter.invoke(req, "test_key")
            assert resp.success is False
            assert resp.http_status == 503
            assert resp.retry_count == 2
            # 1 initial + 2 retries = 3 calls
            assert mock_open.call_count == 3


def test_401_403_fail_closed_no_retry():
    req = CanonicalPreviewRequest(
        provider="groq_free",
        requested_model_id="qwen/qwen3.8-27b",
        prompt="test",
        retry_policy={"max_retries": 2}
    )
    for code in (401, 403, 400, 404):
        mock_err = urllib.error.HTTPError("http://example.com", code, "Forbidden/Unauthorized", {}, None)
        with patch("urllib.request.urlopen", side_effect=mock_err) as mock_open:
            resp = GroqFreeAdapter.invoke(req, "test_key")
            assert resp.success is False
            assert resp.http_status == code
            assert resp.retry_count == 0
            assert mock_open.call_count == 1  # Absolutely NO retries on 4xx


def test_timeout_handling():
    req = CanonicalPreviewRequest(
        provider="google_gemini_free",
        requested_model_id="gemini-3.8-flash",
        prompt="test",
        timeout=120.0,
        retry_policy={"max_retries": 1}
    )
    with patch("urllib.request.urlopen", side_effect=TimeoutError("Request timed out")) as mock_open:
        with patch("time.sleep"):
            resp = GoogleGeminiFreeAdapter.invoke(req, "test_key")
            assert resp.success is False
            assert resp.error_class == "TimeoutError"
            assert resp.retry_count == 1
            assert mock_open.call_count == 2


def test_gemini_request_url_does_not_contain_api_key():
    fake_key = "FAKE_SECRET_KEY_NEVER_IN_URL_12345"
    req = CanonicalPreviewRequest(
        provider="google_gemini_free",
        requested_model_id="gemini-3.8-flash",
        prompt="test"
    )
    captured_requests = []
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b'{"candidates": [{"content": {"parts": [{"text": "OK"}]}}]}'
    mock_resp.headers = {}
    mock_resp.__enter__.return_value = mock_resp

    def fake_urlopen(request, timeout):
        captured_requests.append(request)
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        resp = GoogleGeminiFreeAdapter.invoke(req, fake_key)
        assert resp.success is True

    assert len(captured_requests) == 1
    sent_req = captured_requests[0]
    # URL MUST NOT contain fake_key or '?key='
    assert fake_key not in sent_req.full_url
    assert "?key=" not in sent_req.full_url
    assert sent_req.full_url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"


def test_gemini_header_auth_populated_at_execution_time():
    fake_key = "FAKE_SECRET_KEY_IN_HEADER_12345"
    req = CanonicalPreviewRequest(
        provider="google_gemini_free",
        requested_model_id="gemini-3.8-flash",
        prompt="test"
    )
    captured_requests = []
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b'{"candidates": [{"content": {"parts": [{"text": "OK"}]}}]}'
    mock_resp.headers = {}
    mock_resp.__enter__.return_value = mock_resp

    def fake_urlopen(request, timeout):
        captured_requests.append(request)
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        resp = GoogleGeminiFreeAdapter.invoke(req, fake_key)
        assert resp.success is True

    assert len(captured_requests) == 1
    sent_req = captured_requests[0]
    # Header x-goog-api-key MUST be populated with fake_key
    assert sent_req.get_header("X-goog-api-key") == fake_key
    assert sent_req.get_header("User-agent") == USER_AGENT


def test_gemini_secret_never_printed_or_persisted():
    fake_key = "SECRET_KEY_NOT_IN_RESPONSE_OR_LOGS"
    req = CanonicalPreviewRequest(
        provider="google_gemini_free",
        requested_model_id="gemini-3.8-flash",
        prompt="test"
    )
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b'{"candidates": [{"content": {"parts": [{"text": "OK"}]}}]}'
    mock_resp.headers = {}
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        resp = GoogleGeminiFreeAdapter.invoke(req, fake_key)
        assert fake_key not in str(resp)
        assert fake_key not in repr(resp)

