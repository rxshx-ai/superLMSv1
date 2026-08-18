"""FastAPI service wrapper for the multi-user Moodle LLM Bridge.

This application:
- Starts one background LLMBridgeAgent per configured user (users.json)
- Exposes simple HTTP endpoints for health checks and basic control

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

import agent as agent_module
from agent import LLMBridgeAgent
import config

logger = logging.getLogger("api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-8s | %(levelname)-5s | %(message)s",
)


_agents: list[LLMBridgeAgent] = []
_threads: list[threading.Thread] = []


def start_agents_if_needed() -> None:
    """Ensure a background agent is running for every configured user."""
    global _agents, _threads

    if not _agents:
        logger.info("Initialising user agents for FastAPI service …")
        users = config.validate()
        _agents = agent_module.build_agents(users)

    # (Re)start any agents whose thread isn't alive.
    alive = {t.name for t in _threads if t.is_alive()}
    to_start = [a for a in _agents if f"agent-{a.name}" not in alive]
    if to_start:
        logger.info("Starting %d agent thread(s) …", len(to_start))
        _threads.extend(agent_module.start_agents(to_start))


def stop_agents() -> None:
    """Request every background agent loop to stop."""
    for a in _agents:
        a._running = False


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[override]
    """FastAPI lifespan context to manage the background agents."""
    start_agents_if_needed()
    try:
        yield
    finally:
        stop_agents()


app = FastAPI(
    title="Moodle LLM Bridge API",
    description=(
        "HTTP wrapper around the multi-user Moodle LLM Bridge. "
        "One agent per user polls Moodle for [LLMQ] prompts and posts "
        "[LLMR#<id>] responses using an LLM backend."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["system"])
async def health() -> dict:
    """Basic health check endpoint."""
    return {
        "status": "ok",
        "moodle_url": config.MOODLE_URL,
        "llm_provider": config.LLM_PROVIDER.upper(),
        "users": len(_agents),
        "poll_interval": config.POLL_INTERVAL,
    }


@app.get("/status", tags=["agent"])
async def status() -> dict:
    """Per-user agent status."""
    alive = {t.name for t in _threads if t.is_alive()}
    users = [
        {
            "name": a.name,
            "running": f"agent-{a.name}" in alive,
            "processed_prompts": len(a.processed_ids),
        }
        for a in _agents
    ]
    return {
        "agents_running": sum(1 for u in users if u["running"]),
        "total_users": len(_agents),
        "poll_interval": config.POLL_INTERVAL,
        "users": users,
    }


@app.post("/control/poll-once", tags=["agent"])
async def poll_once() -> dict:
    """Run a single poll cycle for every user (useful for testing)."""
    if not _agents:
        start_agents_if_needed()
    for a in _agents:
        a.poll_once()
    return {"status": "ok", "users_polled": len(_agents)}


@app.post("/control/restart", tags=["agent"])
async def restart_agents() -> dict:
    """Restart all background agent threads."""
    global _agents, _threads
    stop_agents()
    _agents = []
    _threads = []
    start_agents_if_needed()
    return {"status": "restarted", "users": len(_agents)}
