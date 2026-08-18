"""LLM client — AWS Bedrock (via the model-agnostic Converse API) with Groq fallback."""

from __future__ import annotations

import logging
from typing import Optional

import config

logger = logging.getLogger("llm")


_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. The user is asking you a question "
    "through a Moodle LMS dashboard text-block relay system. Provide a clear, well-formatted "
    "response. Use plain text with simple formatting (no markdown headers with #). "
    "Keep the response concise but thorough."
)


class LLMClient:
    """Provider-agnostic LLM wrapper.

    Reads LLM_PROVIDER from config:
        - "bedrock"  → AWS Bedrock via the Converse API (any model:
                       Moonshot Kimi, Anthropic Claude, …), IAM role or keys
        - "groq"     → Groq API (Llama models, fallback)

    Usage:
        client = LLMClient()
        text = client.generate_response("Your prompt")
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        provider = config.LLM_PROVIDER.lower()

        if provider == "bedrock":
            self._backend = _BedrockClient(model_name=model_name)
        elif provider == "groq":
            self._backend = _GroqClient(
                api_key=api_key or config.GROQ_API_KEY,
                model_name=model_name,
            )
        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{provider}'. Set to 'bedrock' or 'groq' in .env"
            )

    def generate_response(self, prompt: str) -> str:
        return self._backend.generate(prompt)


# ── Bedrock Backend ───────────────────────────────────────────────────

class _BedrockClient:
    """Calls a Bedrock model through the provider-agnostic Converse API.

    Converse works uniformly across Bedrock models (Moonshot Kimi,
    Anthropic Claude, Meta Llama, …), so switching models is just a
    config change — no code change.

    Auth (in priority order, handled by boto3's default credential chain):
        1. IAM Role attached to EC2 instance (recommended — no keys needed)
        2. AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY in .env
        3. ~/.aws/credentials on the machine
    """

    def __init__(self, model_name: Optional[str] = None):
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is not installed. Run: pip install boto3"
            )

        self._model = model_name or config.BEDROCK_MODEL

        # If access keys are set in .env, pass them explicitly.
        # If running on EC2 with an IAM role, leave them unset —
        # boto3 picks them up automatically from instance metadata.
        kwargs: dict = {"region_name": config.AWS_REGION}
        if config.AWS_ACCESS_KEY_ID and config.AWS_SECRET_ACCESS_KEY:
            kwargs["aws_access_key_id"] = config.AWS_ACCESS_KEY_ID
            kwargs["aws_secret_access_key"] = config.AWS_SECRET_ACCESS_KEY
            logger.info("Bedrock auth: using explicit access keys from .env")
        else:
            logger.info("Bedrock auth: using IAM role / default credential chain")

        self._client = boto3.client("bedrock-runtime", **kwargs)
        logger.info("Bedrock client ready | model=%s | region=%s",
                    self._model, config.AWS_REGION)

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.converse(
                modelId=self._model,
                system=[{"text": _SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 4096, "temperature": 0.7},
            )
            text = response["output"]["message"]["content"][0]["text"] or ""
            if text:
                logger.info(
                    "Got response (%d chars) [bedrock/%s] for prompt: %.60s…",
                    len(text), self._model, prompt,
                )
                return text
            logger.warning("Empty response from Bedrock for prompt: %.60s…", prompt)
            return "[Agent: Bedrock returned an empty response. Try rephrasing your prompt.]"
        except Exception as e:
            logger.error("Bedrock API error: %s", e)
            return f"[Agent Error: {type(e).__name__} — {e}]"


# ── Groq Backend (fallback) ───────────────────────────────────────────

class _GroqClient:
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError(
                "groq is not installed. Run: pip install groq"
            )

        self._api_key = api_key
        self._model = model_name or getattr(config, "GROQ_MODEL", self.DEFAULT_MODEL)
        self._client = Groq(api_key=self._api_key)
        logger.info("Groq client ready | model=%s", self._model)

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=4096,
            )
            text = response.choices[0].message.content or ""
            if text:
                logger.info(
                    "Got response (%d chars) [groq] for prompt: %.60s…",
                    len(text), prompt,
                )
                return text
            logger.warning("Empty response for prompt: %.60s…", prompt)
            return "[Agent: LLM returned an empty response. Try rephrasing your prompt.]"
        except Exception as e:
            logger.error("Groq API error: %s", e)
            return f"[Agent Error: {type(e).__name__} — {e}]"
