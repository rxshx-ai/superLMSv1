"""
Configuration loader for Moodle LLM Bridge.
Reads settings from .env file or environment variables.

Multi-user: the list of Moodle accounts the bridge services lives in
users.json (see users.example.json). If users.json is absent, the agent
falls back to the single MOODLE_USERNAME / MOODLE_PASSWORD from .env so
older single-user setups keep working unchanged.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Moodle LMS Settings ──────────────────────────────────────────────
# One shared LMS URL for every user. Per-user credentials live in users.json.
MOODLE_URL      = os.getenv("MOODLE_URL", "https://vitolcc.vit.ac.in").rstrip("/")
MOODLE_USERNAME = os.getenv("MOODLE_USERNAME")   # single-user fallback only
MOODLE_PASSWORD = os.getenv("MOODLE_PASSWORD")   # single-user fallback only

USERS_FILE = Path(__file__).parent / "users.json"

# ── LLM Provider ─────────────────────────────────────────────────────
# Set to "bedrock" to use AWS Bedrock, or "groq" for Groq (Llama)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# ── AWS Bedrock Settings ─────────────────────────────────────────────
# Region where Bedrock is available
AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

# Access keys — leave blank when running on EC2 with an IAM role attached.
# boto3 will pick up credentials automatically from instance metadata.
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# Model to use on Bedrock. Invoked through the model-agnostic Converse
# API, so any Bedrock chat model works (Moonshot Kimi, Anthropic Claude, …).
# Copy the exact model / inference-profile ID from the Bedrock console
# → Model catalog. It may carry a region prefix (e.g. "us.moonshotai.kimi-k2.5").
BEDROCK_MODEL = os.getenv("BEDROCK_MODEL", "moonshotai.kimi-k2.5")

# ── Groq Settings (fallback) ─────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Agent Behaviour ──────────────────────────────────────────────────
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))   # seconds
PUBLISH_STATE = os.getenv("PUBLISH_STATE", "draft")     # site | public | draft

# ── Markers ──────────────────────────────────────────────────────────
PROMPT_MARKER   = "[LLMQ]"   # prefix for user prompts
RESPONSE_MARKER = "[LLMR#"   # prefix for agent responses (e.g. [LLMR#142])


# ── Users ────────────────────────────────────────────────────────────

def load_users(users_file: Path = USERS_FILE) -> list[dict]:
    """Return the list of Moodle accounts the bridge should service.

    Each entry is a dict with keys: name, username, password.

    Source priority:
        1. users.json  — a JSON array of {name?, username, password}
        2. MOODLE_USERNAME / MOODLE_PASSWORD from .env (single-user fallback)

    Raises ValueError if users.json exists but is malformed or has an
    entry missing username/password. Returns [] if nothing is configured.
    """
    if users_file.exists():
        try:
            data = json.loads(users_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"{users_file.name} is not valid JSON: {e}") from e

        if not isinstance(data, list):
            raise ValueError(f"{users_file.name} must be a JSON array of user objects.")

        users: list[dict] = []
        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                raise ValueError(f"{users_file.name} entry #{i + 1} must be an object.")
            username = entry.get("username")
            password = entry.get("password")
            name = entry.get("name") or username or f"user{i + 1}"
            if not username or not password:
                raise ValueError(
                    f"{users_file.name} entry '{name}' is missing username or password."
                )
            users.append({"name": str(name), "username": username, "password": password})
        return users

    # Fallback: single user from .env
    username = os.getenv("MOODLE_USERNAME")
    password = os.getenv("MOODLE_PASSWORD")
    if username and password:
        return [{"name": username, "username": username, "password": password}]
    return []


# ── Validation ───────────────────────────────────────────────────────

def validate() -> list[dict]:
    """Validate shared config and return the resolved user list.

    Exits the process on unrecoverable misconfiguration.
    """
    try:
        users = load_users()
    except ValueError as e:
        print(f"User config error: {e}")
        sys.exit(1)

    if not users:
        print(
            "No users configured. Create users.json (see users.example.json) "
            "or set MOODLE_USERNAME / MOODLE_PASSWORD in .env."
        )
        sys.exit(1)

    provider = LLM_PROVIDER.lower()

    if provider == "bedrock":
        # Access keys are optional on EC2 (IAM role handles auth).
        if not AWS_ACCESS_KEY_ID and not AWS_SECRET_ACCESS_KEY:
            print(
                "Bedrock: no explicit AWS keys set — "
                "assuming an IAM role is attached to this instance."
            )
    elif provider == "groq":
        if not GROQ_API_KEY:
            print("Missing required config: GROQ_API_KEY")
            print("Copy .env.example to .env and fill in the values.")
            sys.exit(1)
    else:
        print(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Must be 'bedrock' or 'groq'.")
        sys.exit(1)

    model = BEDROCK_MODEL if provider == "bedrock" else GROQ_MODEL
    print(
        f"Config loaded | LMS: {MOODLE_URL} | Users: {len(users)} | "
        f"Provider: {LLM_PROVIDER.upper()} (model: {model}) | Poll: {POLL_INTERVAL}s"
    )
    return users
