"""
Moodle LLM Bridge — Main Agent

Continuously polls Moodle blog entries tagged as prompts ([LLMQ]),
forwards them to the LLM, and posts the responses back as new blog
entries tagged [LLMR#<id>].

Usage:
    python agent.py
"""

import re
import json
import time
import signal
import logging
import sys
import html
import threading
from pathlib import Path

import config
from moodle_client import MoodleClient, BlogEntry
from llm_client import LLMClient

# ── Logging Setup ─────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-8s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("agent")

# ── State Persistence (per user) ──────────────────────────────────────

def _state_file(name: str) -> Path:
    """Return the processed-ids state file for a given user.

    Each user gets an isolated state file so one user's processed prompts
    never suppress another's.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "default"
    return Path(__file__).parent / f"processed_ids_{safe}.json"


def load_processed_ids(name: str) -> set:
    """Load the set of already-processed blog entry IDs for *name*."""
    state_file = _state_file(name)
    if state_file.exists():
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            return set(data)
        except (json.JSONDecodeError, TypeError):
            return set()
    return set()


def save_processed_ids(name: str, ids: set):
    """Persist the set of processed IDs for *name*."""
    _state_file(name).write_text(
        json.dumps(sorted(ids), indent=2), encoding="utf-8"
    )


# ── Core Logic ────────────────────────────────────────────────────────

class LLMBridgeAgent:
    """
    The main agent that bridges Moodle blog entries <-> LLM.

    Workflow per poll cycle:
      1. Fetch all blog entries for the logged-in user
      2. Identify NEW prompts ([LLMQ] prefix, not yet processed)
      3. For each new prompt:
         a. Extract the actual question
         b. Send to LLM
         c. Post the response as a new blog entry [LLMR#<id>]
         d. Mark the prompt as processed
    """

    def __init__(
        self,
        name: str,
        moodle_url: str,
        username: str,
        password: str,
        llm: LLMClient | None = None,
    ):
        self.name = name
        self.log = logging.getLogger(f"agent[{name}]")

        self.moodle = MoodleClient(
            base_url=moodle_url,
            username=username,
            password=password,
        )
        # The LLM client is safe to share across users/threads; build one
        # per agent only if the caller didn't supply a shared instance.
        self.llm = llm or LLMClient()
        self.processed_ids = load_processed_ids(name)
        self._running = True

    # ── Prompt / Response Identification ──────────────────────────────

    @staticmethod
    def _entry_id(entry: BlogEntry) -> int:
        return entry.entry_id

    @staticmethod
    def _entry_subject(entry: BlogEntry) -> str:
        return entry.subject or ""

    @staticmethod
    def _entry_body(entry: BlogEntry) -> str:
        return entry.body or ""

    @staticmethod
    def _strip_marker(text: str, marker: str) -> str:
        value = (text or "").strip()
        if value.startswith(marker):
            return value.replace(marker, "", 1).strip()
        return value

    @staticmethod
    def is_prompt(entry: BlogEntry) -> bool:
        """Does this blog entry look like a user prompt?"""
        subject = LLMBridgeAgent._entry_subject(entry).strip()
        body    = LLMBridgeAgent._entry_body(entry).strip()
        return (
            subject.startswith(config.PROMPT_MARKER)
            or body.startswith(config.PROMPT_MARKER)
        )

    @staticmethod
    def is_response(entry: BlogEntry) -> bool:
        """Does this blog entry look like an agent response?"""
        return LLMBridgeAgent._entry_subject(entry).strip().startswith(config.RESPONSE_MARKER)

    @staticmethod
    def extract_prompt_text(entry: BlogEntry) -> str:
        """Get the actual question text from a prompt entry."""
        subject_text = LLMBridgeAgent._strip_marker(
            LLMBridgeAgent._entry_subject(entry),
            config.PROMPT_MARKER,
        )
        body_text = LLMBridgeAgent._strip_marker(
            LLMBridgeAgent._entry_body(entry),
            config.PROMPT_MARKER,
        )
        if body_text and len(body_text) > len(subject_text):
            return body_text
        return subject_text or body_text

    def get_response_entry_ids(self, entries: list) -> set:
        """Collect prompt IDs that already have a response blog entry."""
        import re
        answered = set()
        for e in entries:
            m = re.match(r'\[LLMR#(\d+)\]', self._entry_subject(e).strip())
            if m:
                answered.add(int(m.group(1)))
        return answered

    # ── Single Poll Cycle ─────────────────────────────────────────────

    def poll_once(self):
        """Run one scan → process → respond cycle."""
        try:
            entries = self.moodle.get_blog_entries()
        except Exception as e:
            self.log.error("Failed to fetch blog entries: %s", e)
            return

        if not entries:
            self.log.debug("No blog entries found.")
            return

        already_answered = self.get_response_entry_ids(entries)

        new_prompts = [
            e for e in entries
            if self.is_prompt(e)
            and self._entry_id(e) not in self.processed_ids
            and self._entry_id(e) not in already_answered
        ]

        if not new_prompts:
            self.log.debug("No new prompts to process.")
            return

        self.log.info("Found %d new prompt(s) to process.", len(new_prompts))

        for prompt_entry in new_prompts:
            self._process_prompt(prompt_entry)

    def _process_prompt(self, prompt_entry: BlogEntry):
        """Process a single prompt: get LLM response and post it back."""
        prompt_id   = self._entry_id(prompt_entry)
        prompt_text = self.extract_prompt_text(prompt_entry)
        self.log.info(
            "Processing blog entry #%d: %.80s...",
            prompt_id,
            prompt_text,
        )

        # 1. Get LLM response
        try:
            response_text = self.llm.generate_response(prompt_text)
        except Exception as e:
            self.log.error("LLM failed for entry #%d: %s", prompt_id, e)
            response_text = f"[Agent Error: Could not get LLM response - {e}]"

        # 2. Build the response blog entry
        prompt_subject = self._strip_marker(
            self._entry_subject(prompt_entry), config.PROMPT_MARKER
        )
        if not prompt_subject:
            prompt_subject = prompt_text[:80]

        response_subject = f"[LLMR#{prompt_id}] Re: {prompt_subject}"
        response_body    = self._format_response_html(
            prompt_text, response_text, prompt_id
        )

        # 3. Post the response as a new blog entry
        try:
            success = self.moodle.create_blog_entry(
                subject=response_subject,
                body=response_body,
                publish_state=config.PUBLISH_STATE,
            )
            if success:
                self.processed_ids.add(prompt_id)
                save_processed_ids(self.name, self.processed_ids)
                self.log.info(
                    "Posted response blog entry for prompt #%d", prompt_id
                )
            else:
                self.log.error(
                    "Failed to post response for prompt #%d", prompt_id
                )
        except Exception as e:
            self.log.error(
                "Error posting response for entry #%d: %s", prompt_id, e
            )

    @staticmethod
    def _format_response_html(
        prompt_text: str, response_text: str, prompt_id: int
    ) -> str:
        """Format the LLM response as clean HTML for a blog entry."""
        safe_prompt   = html.escape(prompt_text[:500])
        safe_response = html.escape(response_text).replace("\n", "<br>")

        return (
            f'<div style="font-family: sans-serif; line-height: 1.6;">'
            f'<p style="color: #666; font-size: 0.9em;">'
            f'<strong>Your prompt (#{prompt_id}):</strong><br>'
            f'<em>{safe_prompt}</em></p>'
            f'<hr style="border: 1px solid #ddd;">'
            f'<div style="margin-top: 10px;">'
            f'{safe_response}'
            f'</div>'
            f'</div>'
        )

    # ── Main Loop ─────────────────────────────────────────────────────

    def run(self):
        """Start the continuous polling loop for this user.

        A login failure stops *this* user's loop only — it never exits the
        process, so other users' agents keep running.
        """
        self.log.info("Agent starting for user '%s'", self.name)

        # Signal handlers can only be registered from the main thread.
        # When many agents run in worker threads, the parent process wires
        # up shutdown instead (see run_all / the FastAPI service).
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._shutdown)
            signal.signal(signal.SIGTERM, self._shutdown)

        try:
            self.moodle.login()
        except Exception as e:
            self.log.critical("Could not log in to Moodle: %s — agent stopped.", e)
            self._running = False
            return

        self.log.info(
            "Polling every %ds for [LLMQ] blog entries", config.POLL_INTERVAL
        )

        while self._running:
            self.poll_once()
            if self._running:
                time.sleep(config.POLL_INTERVAL)

        self.log.info("Agent stopped for user '%s'.", self.name)

    def _shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        self.log.info("Received signal %s - shutting down", signum)
        self._running = False


# ── Multi-user Runner ─────────────────────────────────────────────────

def build_agents(users: list[dict], llm: LLMClient | None = None) -> list["LLMBridgeAgent"]:
    """Create one agent per user, all sharing a single LLM client."""
    shared_llm = llm or LLMClient()
    return [
        LLMBridgeAgent(
            name=u["name"],
            moodle_url=config.MOODLE_URL,
            username=u["username"],
            password=u["password"],
            llm=shared_llm,
        )
        for u in users
    ]


def start_agents(agents: list["LLMBridgeAgent"]) -> list[threading.Thread]:
    """Start each agent's poll loop in its own daemon thread."""
    threads: list[threading.Thread] = []
    for agent in agents:
        agent._running = True
        t = threading.Thread(target=agent.run, name=f"agent-{agent.name}", daemon=True)
        t.start()
        threads.append(t)
    return threads


def run_all() -> None:
    """Validate config, then run one poll loop per configured user."""
    users = config.validate()
    agents = build_agents(users)
    threads = start_agents(agents)

    logger.info("Started %d user agent(s).", len(agents))

    def _shutdown(signum, frame):
        logger.info("Received signal %s - stopping all agents", signum)
        for a in agents:
            a._running = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        for a in agents:
            a._running = False

    logger.info("All agents stopped.")


# ── Entry Point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    run_all()
