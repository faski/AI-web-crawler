"""HTTP client for the backend API.

All communication with the backend goes through this module.
BACKEND_URL defaults to localhost:8003 and is overridden via the
BACKEND_URL environment variable (set in docker-compose.yaml for Docker).
"""

import os

import requests

BACKEND = os.environ.get("BACKEND_URL", "http://127.0.0.1:8003")

SHORT_TIMEOUT = 5
PARSE_TIMEOUT = 60
JUDGE_TIMEOUT = 240


class BackendUnavailable(Exception):
    """Raised when the backend cannot be reached."""


# ─── Read ────────────────────────────────────────────────────────────────────

def get_status() -> dict:
    """Return the per-component status (backend/database/ollama)."""
    try:
        return requests.get(f"{BACKEND}/status", timeout=SHORT_TIMEOUT).json()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as error:
        raise BackendUnavailable() from error


def get_domains() -> list[str]:
    """Return the list of supported domains."""
    try:
        return requests.get(f"{BACKEND}/domains", timeout=SHORT_TIMEOUT).json().get("domains", [])
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as error:
        raise BackendUnavailable() from error


def get_gs_urls(domain: str) -> list[str]:
    """Return all gold standard URLs for a domain."""
    try:
        response = requests.get(
            f"{BACKEND}/gold_standard_urls",
            params={"domain": domain},
            timeout=SHORT_TIMEOUT,
        )
        return response.json().get("gold_standard_urls", [])
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as error:
        raise BackendUnavailable() from error


def get_gold_standard(url: str) -> dict | None:
    """Return the gold standard entry (with html_text + gold_text) for a URL."""
    try:
        response = requests.get(
            f"{BACKEND}/gold_standard",
            params={"url": url},
            timeout=SHORT_TIMEOUT,
        )
        return response.json() if response.status_code == 200 else None
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as error:
        raise BackendUnavailable() from error


def get_db_stats() -> dict:
    """Return per-domain counts and average evaluations."""
    try:
        return requests.get(f"{BACKEND}/db_stats", timeout=SHORT_TIMEOUT).json()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as error:
        raise BackendUnavailable() from error


# ─── Parse + evaluate ────────────────────────────────────────────────────────

def parse_url(url: str, local: bool = False) -> tuple[dict, str | None]:
    """Parse a URL via the backend. Returns ``(data, None)`` or ``({}, error_message)``."""
    try:
        response = requests.post(
            f"{BACKEND}/parse",
            json={"url": url, "local": local},
            timeout=PARSE_TIMEOUT,
        )
        if response.status_code != 200:
            return {}, response.json().get("detail", f"Backend error {response.status_code}")
        return response.json(), None
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as error:
        raise BackendUnavailable() from error


def evaluate(parsed_text: str, gold_text: str) -> dict | None:
    """Compute quantitative evaluation metrics."""
    try:
        response = requests.post(
            f"{BACKEND}/evaluate",
            json={"parsed_text": parsed_text, "gold_text": gold_text},
            timeout=SHORT_TIMEOUT,
        )
        return response.json() if response.status_code == 200 else None
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as error:
        raise BackendUnavailable() from error


def evaluate_judge(parsed_text: str, gold_text: str) -> dict | None:
    """Compute the LLM judge score and feedback."""
    try:
        response = requests.post(
            f"{BACKEND}/evaluate_judge",
            json={"parsed_text": parsed_text, "gold_text": gold_text},
            timeout=JUDGE_TIMEOUT,
        )
        return response.json() if response.status_code == 200 else None
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as error:
        raise BackendUnavailable() from error


# ─── Write + delete ──────────────────────────────────────────────────────────

def add_web_resource(url: str, html_text: str) -> tuple[bool, str | None]:
    """Insert a web_resources row. Returns ``(success, error_message_or_None)``."""
    try:
        response = requests.post(
            f"{BACKEND}/add_web_resource",
            json={"url": url, "html_text": html_text},
            timeout=SHORT_TIMEOUT,
        )
        if response.status_code == 200:
            return True, None
        return False, response.json().get("detail", f"Backend error {response.status_code}")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as error:
        raise BackendUnavailable() from error


def add_gold_standard(url: str, gold_text: str) -> tuple[bool, str | None]:
    """Insert a gold_standard row. Returns ``(success, error_message_or_None)``."""
    try:
        response = requests.post(
            f"{BACKEND}/add_gold_standard",
            json={"url": url, "gold_text": gold_text},
            timeout=SHORT_TIMEOUT,
        )
        if response.status_code == 200:
            return True, None
        return False, response.json().get("detail", f"Backend error {response.status_code}")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as error:
        raise BackendUnavailable() from error


def delete_gold_standard(url: str) -> tuple[bool, str | None]:
    """Remove a gold_standard row. Returns ``(success, error_message_or_None)``."""
    try:
        response = requests.delete(
            f"{BACKEND}/gold_standard",
            json={"url": url},
            timeout=SHORT_TIMEOUT,
        )
        if response.status_code == 200:
            return True, None
        return False, response.json().get("detail", f"Backend error {response.status_code}")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as error:
        raise BackendUnavailable() from error
