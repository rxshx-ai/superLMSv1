"""
Configuration loader for Moodle LLM Bridge.
Reads settings from .env file or environment variables.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# ── Moodle LMS Settings ──────────────────────────────────────────────
MOODLE_URL      = os.getenv("MOODLE_URL", "https://vitolcc.vit.ac.in").rstrip("/")
MOODLE_USERNAME = os.getenv("MOODLE_USERNAME")
MOODLE_PASSWORD = os.getenv("MOODLE_PASSWORD")

# ── LLM Provider ─────────────────────────────────────────────────────
# Set to "bedrock" to use AWS Bedrock (Claude), or "groq" for Groq (Llama)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "bedrock")

# ── AWS Bedrock Settings ─────────────────────────────────────────────
# Region where Bedrock is available
AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

# Access keys — leave blank when running on EC2 with an IAM role attached.
# The SDK will pick up credentials automatically from instance metadata.
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# Claude model to use on Bedrock.
# Options:
#   us.anthropic.claude-3-5-sonnet-20241022-v2:0  ← recommended (fast + capable)
#   us.anthropic.claude-3-opus-20240229-v1:0       ← most powerful
#   us.anthropic.claude-3-haiku-20240307-v1:0      ← fastest + cheapest
BEDROCK_MODEL = os.getenv(
    "BEDROCK_MODEL",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
)

# ── Groq Settings (fallback) ─────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Agent Behaviour ──────────────────────────────────────────────────
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))   # seconds
PUBLISH_STATE = os.getenv("PUBLISH_STATE", "draft")     # site | public | draft

# ── Markers ──────────────────────────────────────────────────────────
PROMPT_MARKER   = "[LLMQ]"   # prefix for user prompts
RESPONSE_MARKER = "[LLMR#"   # prefix for agent responses (e.g. [LLMR#142])


# ── Validation ───────────────────────────────────────────────────────

def validate():
    """Check that all required configuration values are set."""
    errors = []

    if not MOODLE_USERNAME:
        errors.append("MOODLE_USERNAME")
    if not MOODLE_PASSWORD:
        errors.append("MOODLE_PASSWORD")

    provider = LLM_PROVIDER.lower()

    if provider == "bedrock":
        # Access keys are optional on EC2 (IAM role handles auth).
        # Warn if neither keys nor a likely IAM environment are detected.
        if not AWS_ACCESS_KEY_ID and not AWS_SECRET_ACCESS_KEY:
            # This is fine on EC2 — just log it, don't fail.
            print(
                "Bedrock: no explicit AWS keys set — "
                "assuming IAM role is attached to this instance."
            )
    elif provider == "groq":
        if not GROQ_API_KEY:
            errors.append("GROQ_API_KEY")
    else:
        print(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Must be 'bedrock' or 'groq'.")
        sys.exit(1)

    if errors:
        print(f"Missing required config: {', '.join(errors)}")
        print("Copy .env.example to .env and fill in the values.")
        sys.exit(1)

    print(
        f"Config loaded | LMS: {MOODLE_URL} | "
        f"Provider: {LLM_PROVIDER.upper()} "
        f"({'model: ' + BEDROCK_MODEL if provider == 'bedrock' else 'model: ' + GROQ_MODEL}) | "
        f"Poll: {POLL_INTERVAL}s"
    )
