"""HTTP client for the Ollama API.

Configuration is read from environment variables:
    OLLAMA_HOST   base URL    (default: http://localhost:11434)
    OLLAMA_MODEL  model name  (default: llama3.2:3b)
"""

import os

import requests

GENERATE_TIMEOUT_SECONDS = 180.0
PING_TIMEOUT_SECONDS = 5.0


def _llm_config() -> dict:
    """Read OLLAMA_HOST and OLLAMA_MODEL from the environment."""
    return {
        "host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        "model": os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
    }


def get_model_name() -> str:
    """Return the model name currently configured for the judge."""
    return _llm_config()["model"]


def generate(prompt: str, response_format: dict | None = None) -> str:
    """Send a prompt to Ollama and return the raw text response.

    If ``response_format`` (a JSON schema) is given, Ollama forces the answer
    to match it, so the reply is always a complete, valid JSON object.
    """
    config = _llm_config()
    payload = {
        "model": config["model"],
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": -1,
        "options": {"temperature": 0, "num_predict": 1024},
    }
    if response_format is not None:
        payload["format"] = response_format
    response = requests.post(
        f"{config['host']}/api/generate",
        json=payload,
        timeout=GENERATE_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["response"]


def ping() -> bool:
    """Return True if the Ollama server responds, False otherwise."""
    try:
        config = _llm_config()
        response = requests.get(f"{config['host']}/api/tags", timeout=PING_TIMEOUT_SECONDS)
        return response.status_code == 200
    except requests.RequestException:
        return False
