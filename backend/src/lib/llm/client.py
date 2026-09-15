"""HTTP client for the LLM backends.

Two providers sit behind the same ``generate()`` call, chosen with the
``LLM_PROVIDER`` environment variable:

    ollama      (default) the local Ollama server used by the base project
    openrouter  OpenRouter, for the models that do not fit in local RAM

Configuration is read from environment variables:

    LLM_PROVIDER         ollama | openrouter  (default: ollama)
    LLM_TIMEOUT_SECONDS  request timeout in seconds (default: 180)

    OLLAMA_HOST          base URL    (default: http://localhost:11434)
    OLLAMA_MODEL         model name  (default: qwen3.5:4b)
    OLLAMA_NUM_CTX       context window in tokens (default: Ollama's own)

    OPENROUTER_API_KEY   required when LLM_PROVIDER=openrouter
    OPENROUTER_MODEL     model id    (default: qwen/qwen3.5-9b)
    OPENROUTER_PROVIDER  pin one upstream provider, e.g. "DeepInfra" (optional)

    LLM_EUR_PER_USD      euro per dollar used to report cost (default: 0.92)

With ``LLM_PROVIDER`` unset the request sent to Ollama is exactly the one this
module sent before it grew a second provider, so the path the grader exercises
is unchanged.
"""

import os
import time
from dataclasses import dataclass

import requests

# Euro per dollar used to report cost. Providers bill in dollars, so the
# dollar figure is the fact and the euro figure is a conversion; the rate is
# configurable and recorded with each run so an old number stays readable.
DEFAULT_EUR_PER_USD = 0.92

DEFAULT_TIMEOUT_SECONDS = 180.0
PING_TIMEOUT_SECONDS = 5.0

# Enough for a judge verdict. The parser path asks for more via max_tokens,
# since there the model has to emit the whole page as markdown.
DEFAULT_MAX_TOKENS = 1024

OPENROUTER_HOST = "https://openrouter.ai/api/v1"

# A long answer can have its connection dropped mid-body, and a busy
# provider answers 429 or 503. Both are transient and worth retrying: a
# whole page would otherwise be lost to a hiccup that lasts a second.
DEFAULT_MAX_ATTEMPTS = 4
RETRY_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# Ollama also answers 404 "model not found" when the model is there but its
# blob could not be opened. On macOS the gRPC FUSE mount that backs
# ollama_data occasionally reports ENOENT for a file that exists, and the next
# attempt succeeds. That failure mode is local to Ollama: a 404 from
# OpenRouter means the model id is wrong, and retrying it is pointless.
OLLAMA_RETRY_STATUS_CODES = RETRY_STATUS_CODES | {404}
TRANSIENT_ERRORS = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def _provider() -> str:
    """Return the active provider name, lowercased."""
    return os.environ.get("LLM_PROVIDER", "ollama").strip().lower()


def _timeout() -> float:
    """Return the request timeout in seconds."""
    return float(os.environ.get("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))


def get_model_name() -> str:
    """Return the model name currently configured for the active provider.

    This value is stored alongside every result, so runs made with different
    models stay distinguishable in the database.
    """
    if _provider() == "openrouter":
        return os.environ.get("OPENROUTER_MODEL", "qwen/qwen3.5-9b")
    return os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")


@dataclass(frozen=True)
class Usage:
    """What one model call consumed.

    ``cost_usd`` is what the provider says it charged, not a figure computed
    from a price list: prices change, promotions apply, and a provider may
    bill cached input differently. Ollama runs on the machine and charges
    nothing, so it reports zero - which is true, and the CPU seconds measured
    elsewhere are where its cost shows up instead.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


def eur_per_usd() -> float:
    """Return the euro-per-dollar rate used to report costs."""
    return float(os.environ.get("LLM_EUR_PER_USD", DEFAULT_EUR_PER_USD))


def generate(
    prompt: str,
    response_format: dict | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Send a prompt to the active provider and return the raw text response.

    Args:
        prompt:          the full prompt to send.
        response_format: a JSON schema. When given, the provider is asked to
            force the answer to match it, so the reply is a complete, valid
            JSON object.
        max_tokens:      cap on the generated length. The default suits the
            judge; a caller that needs a whole page of markdown must raise it.

    Raises:
        RuntimeError: if the provider name is unknown, or the OpenRouter key
            is missing.
        requests.HTTPError: if the provider returns an error status.
    """
    return generate_with_usage(prompt, response_format, max_tokens)[0]


def generate_with_usage(
    prompt: str,
    response_format: dict | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[str, Usage]:
    """Send a prompt and return the text together with what the call consumed.

    Same call as ``generate``; the difference is that the accounting comes
    back instead of being thrown away. A caller that reads a page in several
    fragments needs it per fragment, because "what did this page cost" is the
    sum of its calls and not something that can be reconstructed afterwards -
    the provider reports a running total, not a per-request breakdown.
    """
    if _provider() == "openrouter":
        return _generate_openrouter(prompt, response_format, max_tokens)
    if _provider() == "ollama":
        return _generate_ollama(prompt, response_format, max_tokens)
    raise RuntimeError(f"Unknown LLM_PROVIDER: {_provider()!r}")


def _max_attempts() -> int:
    """Return how many times a request may be sent before giving up."""
    return int(os.environ.get("LLM_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS))


def _post_json(
    url: str,
    payload: dict,
    headers: dict | None = None,
    retry_statuses: frozenset = RETRY_STATUS_CODES,
) -> dict:
    """POST and return the decoded JSON body, retrying transient failures.

    Reading the body is part of the retried block on purpose: a dropped
    connection surfaces while the response is being consumed, not when it is
    opened, so retrying only the request itself would miss the common case.

    Args:
        retry_statuses: HTTP codes worth another attempt. The caller decides,
            because the same code means different things per provider.
    """
    attempts = _max_attempts()
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(
                url, json=payload, headers=headers, timeout=_timeout()
            )
            if response.status_code in retry_statuses and attempt < attempts:
                raise requests.HTTPError(
                    f"{response.status_code} from provider", response=response
                )
            response.raise_for_status()
            return response.json()
        except (*TRANSIENT_ERRORS, requests.HTTPError, ValueError) as error:
            fatal = isinstance(error, requests.HTTPError) and (
                error.response is not None
                and error.response.status_code not in retry_statuses
            )
            if fatal or attempt == attempts:
                raise
            delay = 2 ** attempt
            print(
                f"[llm] tentativo {attempt}/{attempts} fallito "
                f"({type(error).__name__}), riprovo fra {delay}s",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")


# --- Ollama ------------------------------------------------------------------


def _ollama_config() -> dict:
    """Read the Ollama host and model from the environment."""
    return {
        "host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        "model": os.environ.get("OLLAMA_MODEL", "qwen3.5:4b"),
    }


def _ollama_options(max_tokens: int) -> dict:
    """Build the Ollama options block.

    ``num_ctx`` is only sent when OLLAMA_NUM_CTX is set, so the default request
    stays identical to the one the base project used. Setting it matters for
    long inputs: Ollama silently drops whatever does not fit in the context
    window instead of reporting an error, so a prompt longer than the default
    window is truncated without any warning.
    """
    options = {"temperature": 0, "num_predict": max_tokens}
    num_ctx = os.environ.get("OLLAMA_NUM_CTX")
    if num_ctx:
        options["num_ctx"] = int(num_ctx)
    return options


def _generate_ollama(
    prompt: str, response_format: dict | None, max_tokens: int
) -> tuple[str, Usage]:
    """Call the Ollama /api/generate endpoint and return the text and usage."""
    config = _ollama_config()
    payload = {
        "model": config["model"],
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": -1,
        "options": _ollama_options(max_tokens),
    }
    if response_format is not None:
        payload["format"] = response_format
    body = _post_json(
        f"{config['host']}/api/generate",
        payload,
        retry_statuses=OLLAMA_RETRY_STATUS_CODES,
    )
    return body["response"], Usage(
        prompt_tokens=int(body.get("prompt_eval_count") or 0),
        completion_tokens=int(body.get("eval_count") or 0),
        cost_usd=0.0,
    )


# --- OpenRouter --------------------------------------------------------------


def _openrouter_key() -> str:
    """Return the OpenRouter API key, or explain that it is missing."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set, but LLM_PROVIDER=openrouter."
        )
    return key


def _openrouter_response_format(response_format: dict | None) -> dict | None:
    """Wrap a bare JSON schema in the OpenAI-compatible envelope.

    Ollama takes the schema directly; the OpenAI-style API expects it nested
    under ``json_schema``. ``strict`` is left off on purpose: the open-weight
    providers implement it unevenly and would reject the request outright,
    while the judge already copes with a slightly malformed reply.
    """
    if response_format is None:
        return None
    return {
        "type": "json_schema",
        "json_schema": {"name": "result", "schema": response_format},
    }


def _openrouter_payload(
    prompt: str, response_format: dict | None, max_tokens: int
) -> dict:
    """Build the OpenRouter chat-completions payload."""
    payload = {
        "model": get_model_name(),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        # Extraction gains nothing from an explicit reasoning trace, which
        # would cost output tokens and blur the comparison between models
        # that have the feature and models that do not.
        "reasoning": {"enabled": False},
        # Without this the reply carries no cost, and what a page cost could
        # then only be guessed from a price list.
        "usage": {"include": True},
    }
    schema = _openrouter_response_format(response_format)
    if schema is not None:
        payload["response_format"] = schema

    # Pinning the upstream provider keeps the served quantisation stable
    # across a run; without it the same model id can be answered by providers
    # serving different weights, which would confound a comparison of sizes.
    provider = os.environ.get("OPENROUTER_PROVIDER", "").strip()
    if provider:
        payload["provider"] = {"order": [provider], "allow_fallbacks": False}
    return payload


def _generate_openrouter(
    prompt: str, response_format: dict | None, max_tokens: int
) -> tuple[str, Usage]:
    """Call OpenRouter chat-completions and return the text and usage."""
    body = _post_json(
        f"{OPENROUTER_HOST}/chat/completions",
        _openrouter_payload(prompt, response_format, max_tokens),
        headers={"Authorization": f"Bearer {_openrouter_key()}"},
    )
    # OpenRouter reports upstream failures as a 200 carrying an error body.
    if "error" in body and "choices" not in body:
        raise RuntimeError(f"OpenRouter error: {body['error']}")
    usage = body.get("usage") or {}
    return body["choices"][0]["message"]["content"] or "", Usage(
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        cost_usd=float(usage.get("cost") or 0.0),
    )


# --- Health check ------------------------------------------------------------


def ping() -> bool:
    """Return True if the active provider responds, False otherwise."""
    try:
        if _provider() == "openrouter":
            response = requests.get(
                f"{OPENROUTER_HOST}/key",
                headers={"Authorization": f"Bearer {_openrouter_key()}"},
                timeout=PING_TIMEOUT_SECONDS,
            )
        else:
            config = _ollama_config()
            response = requests.get(
                f"{config['host']}/api/tags", timeout=PING_TIMEOUT_SECONDS
            )
        return response.status_code == 200
    except (requests.RequestException, RuntimeError):
        return False
