"""Tests for the Bedrock backend — verifies it calls the Converse API correctly."""

import importlib
from unittest.mock import MagicMock, patch

import config
llm_client = importlib.import_module("llm_client")


def _fake_converse_response(text: str) -> dict:
    return {"output": {"message": {"content": [{"text": text}]}}}


def test_bedrock_uses_converse_and_returns_text(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "bedrock")
    monkeypatch.setattr(config, "BEDROCK_MODEL", "moonshotai.kimi-k2.5")
    monkeypatch.setattr(config, "AWS_ACCESS_KEY_ID", "")
    monkeypatch.setattr(config, "AWS_SECRET_ACCESS_KEY", "")

    fake_client = MagicMock()
    fake_client.converse.return_value = _fake_converse_response("hello from kimi")

    with patch("boto3.client", return_value=fake_client) as mk:
        client = llm_client.LLMClient()
        out = client.generate_response("hi there")

    assert out == "hello from kimi"
    mk.assert_called_once()
    assert mk.call_args.args[0] == "bedrock-runtime"

    # The Converse call must carry the configured model and the prompt.
    kwargs = fake_client.converse.call_args.kwargs
    assert kwargs["modelId"] == "moonshotai.kimi-k2.5"
    assert kwargs["messages"] == [
        {"role": "user", "content": [{"text": "hi there"}]}
    ]
    assert kwargs["system"][0]["text"]  # a system prompt is sent


def test_bedrock_api_error_is_caught(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "bedrock")

    fake_client = MagicMock()
    fake_client.converse.side_effect = RuntimeError("throttled")

    with patch("boto3.client", return_value=fake_client):
        client = llm_client.LLMClient()
        out = client.generate_response("hi")

    assert out.startswith("[Agent Error:")
    assert "throttled" in out
