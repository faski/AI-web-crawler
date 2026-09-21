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
module sent before it grew a second provider, so the path the grader uses is
unchanged.
"""

import os
import time
from dataclasses import dataclass

import requests

# Euro per dollar used to report cost. Providers bill in dollars, so that is
# the fact and the euro figure is a conversion. The rate is recorded with each
# run, so an old number can still be read.
DEFAULT_EUR_PER_USD = 0.92

DEFAULT_TIMEOUT_SECONDS = 180.0
PING_TIMEOUT_SECONDS = 5.0

# Enough for a judge verdict. The parser path asks for more via max_tokens,
# since there the model has to emit the whole page as markdown.
DEFAULT_MAX_TOKENS = 1024

OPENROUTER_HOST = "https://openrouter.ai/api/v1"

# A long answer can lose its connection mid-body, and a busy provider answers
# 429 or 503. Both pass, and retrying saves a whole page from a one-second
# hiccup.
DEFAULT_MAX_ATTEMPTS = 4
RETRY_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# Ollama also answers 404 "model not found" when the model is there but its
# blob could not be opened: on macOS the gRPC FUSE mount under ollama_data
# sometimes reports ENOENT for a file that exists, and the next try works.
# Only Ollama. A 404 from OpenRouter means the model id is wrong.
OLLAMA_RETRY_STATUS_CODES = RETRY_STATUS_CODES | {404}
TRANSIENT_ERRORS = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def env_number(name: str, default, convert=float):
    """Read a numeric setting, treating an empty value as not set.

    docker-compose writes an optional setting as ``VAR: ${VAR:-}``, which puts
    an empty string in the environment instead of leaving it out. float("")
    raises, so a setting nobody chose would crash the request instead of
    falling back to its default.
    """
    raw = os.environ.get(name, "").strip()
    return convert(raw) if raw else default


def _provider() -> str:
    """Return the active provider name, lowercased."""
    return os.environ.get("LLM_PROVIDER", "ollama").strip().lower()


def _timeout() -> float:
    """Return the request timeout in seconds."""
    return env_number("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)


def get_provider() -> str:
    """Return the active provider name: "ollama" or "openrouter".

    Public because the pages have to say who will answer. Whether a call
    costs money depends on this alone, and a page that reads it from a
    constant instead will sooner or later say the wrong thing.
    """
    return _provider()


def get_model_name() -> str:
    """Return the model name configured for the active provider.

    Stored with every result, so runs made with different models stay
    distinguishable in the database.
    """
    if _provider() == "openrouter":
        return os.environ.get("OPENROUTER_MODEL", "qwen/qwen3.5-9b")
    return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


@dataclass(frozen=True)
class Usage:
    """What one model call consumed.

    ``cost_usd`` is what the provider says it charged, not a figure from a
    price list: prices change and cached input can be billed differently.
    Ollama runs here and charges nothing, so it reports zero; its cost shows
    up as the CPU seconds measured elsewhere.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    # Which upstream provider answered, and the id it filed the call under.
    # OpenRouter is a router, not a host: the same model id is served by
    # several companies at different quantisations - fp4, fp8 and bf16 were
    # all on offer for qwen3.5-9b - and without pinning it picks one per
    # request. A run that does not record this cannot say which weights
    # produced its numbers. Empty on Ollama, which has no upstream to name.
    provider: str = ""
    generation_id: str = ""
    # Why the model stopped writing. "length" means it hit max_tokens and the
    # answer is cut mid-word; anything else means it finished by itself. From
    # outside the two look the same - both return a non-empty string - so
    # without this a truncated reply is read as a badly formatted one.
    finish_reason: str = ""


def eur_per_usd() -> float:
    """Return the euro-per-dollar rate used to report costs."""
    return env_number("LLM_EUR_PER_USD", DEFAULT_EUR_PER_USD)


def generate(
    prompt: str,
    response_format: dict | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    model: str | None = None,
    num_ctx: int | None = None,
) -> str:
    """Send a prompt to the active provider and return the raw text response.

    Args:
        prompt:          the full prompt to send.
        response_format: a JSON schema. When given, the provider is asked to
            force the answer to match it, so the reply is a complete, valid
            JSON object.
        max_tokens:      cap on the generated length. The default suits the
            judge; a caller that needs a whole page of markdown must raise it.
        model:           use this local model instead of the configured one.
            Ignored on OpenRouter, where there is a single configured model.
        num_ctx:         local context window to ask for. 0 means "let Ollama
            choose"; omitted means "whatever OLLAMA_NUM_CTX says".

    Raises:
        RuntimeError: if the provider name is unknown, or the OpenRouter key
            is missing.
        requests.HTTPError: if the provider returns an error status.
    """
    return generate_with_usage(prompt, response_format, max_tokens, model, num_ctx)[0]


def generate_with_usage(
    prompt: str,
    response_format: dict | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    model: str | None = None,
    num_ctx: int | None = None,
) -> tuple[str, Usage]:
    """Send a prompt and return the text together with what the call consumed.

    Same call as ``generate``, except the accounting comes back instead of
    being thrown away. A caller that reads a page in several fragments needs
    it per fragment: what the page cost is the sum of its calls, and the
    provider only reports a running total afterwards.
    """
    if _provider() == "openrouter":
        return _generate_openrouter(prompt, response_format, max_tokens)
    if _provider() == "ollama":
        return _generate_ollama(prompt, response_format, max_tokens, model, num_ctx)
    raise RuntimeError(f"Unknown LLM_PROVIDER: {_provider()!r}")


def _max_attempts() -> int:
    """Return how many times a request may be sent before giving up."""
    return env_number("LLM_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS, int)


def _post_json(
    url: str,
    payload: dict,
    headers: dict | None = None,
    retry_statuses: frozenset = RETRY_STATUS_CODES,
) -> dict:
    """POST and return the decoded JSON body, retrying transient failures.

    Reading the body is inside the retried block on purpose: a dropped
    connection shows up while the response is being read, not when it is
    opened, so retrying only the request would miss the common case.

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
            if not response.ok:
                # raise_for_status() keeps the status and throws the body
                # away, but the body is where the provider explains itself:
                # OpenRouter answers 400 with "Reasoning is mandatory for this
                # endpoint", and without it you only see "400 Client Error".
                raise requests.HTTPError(
                    f"{response.status_code} from provider: {response.text[:300]}",
                    response=response,
                )
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


DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"

# The judge scores extractions, the parser produces them, and on a machine
# this size they should not be the same model: the judge runs on every load of
# /parser and wants the smallest model that can give a verdict, the parser
# wants the biggest that fits. Its own default, so the judge's scores stay
# comparable with the ones the base project recorded.
DEFAULT_JUDGE_MODEL = "llama3.2:3b"


def get_judge_model_name() -> str:
    """Return the model the judge should use, for the active provider.

    On OpenRouter there is one model and the judge shares it. Locally it has
    its own, because the two jobs want opposite things.
    """
    if _provider() == "openrouter":
        return get_model_name()
    return os.environ.get("OLLAMA_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


def _ollama_config(model: str | None = None) -> dict:
    """Read the Ollama host and model from the environment.

    ``model`` overrides the configured one, for a caller that needs something
    other than the model the stack parses with.
    """
    return {
        "host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        "model": model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
    }


def _ollama_options(max_tokens: int, num_ctx: int | None = None) -> dict:
    """Build the Ollama options block.

    ``num_ctx`` is only sent when asked for, so the default request stays the
    one the base project used. It matters for long inputs: Ollama drops
    whatever does not fit in the window instead of reporting an error, so a
    long prompt is truncated without warning.

    It also costs memory, because the window is allocated whether the prompt
    fills it or not. A caller with a short prompt must not inherit the
    parser's setting: here that turned a 2.9 GB judge model into a 4.4 GB one
    for a prompt under a thousand tokens.
    """
    options = {"temperature": 0, "num_predict": max_tokens}
    configured = num_ctx if num_ctx is not None else os.environ.get("OLLAMA_NUM_CTX")
    if configured:
        options["num_ctx"] = int(configured)
    return options


def _generate_ollama(
    prompt: str,
    response_format: dict | None,
    max_tokens: int,
    model: str | None = None,
    num_ctx: int | None = None,
) -> tuple[str, Usage]:
    """Call the Ollama /api/generate endpoint and return the text and usage."""
    config = _ollama_config(model)
    payload = {
        "model": config["model"],
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": _ollama_options(max_tokens, num_ctx),
    }
    keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE")
    if keep_alive:
        payload["keep_alive"] = keep_alive
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
        # Ollama calls it done_reason, same two values.
        finish_reason=str(body.get("done_reason") or ""),
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

    Ollama takes the schema directly; the OpenAI-style API wants it nested
    under ``json_schema``. ``strict`` is left off on purpose: the open-weight
    providers support it unevenly and would reject the request, and the judge
    already copes with a slightly malformed reply.
    """
    if response_format is None:
        return None
    return {
        "type": "json_schema",
        "json_schema": {"name": "result", "schema": response_format},
    }


def _reasoning_allowed() -> bool:
    """Return whether the model may spend output tokens on a reasoning trace."""
    return os.environ.get("LLM_ALLOW_REASONING", "").strip().lower() in {
        "1",
        "true",
        "yes",
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
        # Without this the reply carries no cost, and what a page cost could
        # only be guessed from a price list.
        "usage": {"include": True},
    }

    # Extraction gains nothing from a reasoning trace, which costs output
    # tokens and blurs the comparison between models that have the feature and
    # models that do not. Some refuse to answer without it ("Reasoning is
    # mandatory for this endpoint", HTTP 400): comparing one of those means
    # LLM_ALLOW_REASONING=1 and saying so next to the result, because that run
    # is not measuring the same thing.
    if not _reasoning_allowed():
        payload["reasoning"] = {"enabled": False}

    schema = _openrouter_response_format(response_format)
    if schema is not None:
        payload["response_format"] = schema

    # Pinning the upstream provider keeps the quantisation stable across a
    # run. Without it the same model id can be answered by providers serving
    # different weights, which would confound any comparison.
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
    choice = body["choices"][0]
    content = choice["message"].get("content") or ""

    # An empty answer used to come back as an empty string, and the caller
    # scored it as a page with nothing in it: a zero that looks like a bad
    # model instead of a failed call. The usual cause is a reasoning model
    # spending the whole budget on the reasoning field and never starting the
    # answer, so say so instead of recording a silent zero.
    if not content.strip():
        reasoning = choice["message"].get("reasoning") or ""
        if reasoning.strip():
            detail = (
                f"{len(reasoning)} characters of reasoning and no answer; "
                "raise max_tokens or use a model that does not reason"
            )
        else:
            detail = f"finish_reason={choice.get('finish_reason')!r}"
        raise RuntimeError(f"empty answer from {get_model_name()}: {detail}")

    return content, Usage(
        provider=str(body.get("provider") or ""),
        generation_id=str(body.get("id") or ""),
        finish_reason=str(choice.get("finish_reason") or ""),
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
