"""LLM-as-Judge: evaluate parser output quality using an Ollama-hosted model.

Modules:
  client    HTTP client for the Ollama API (generate + ping)
  prompt    builds the judge prompt from parsed_text and gold_text
  judge     high-level wrapper with JSON fallback handling
  models    Pydantic model for the judge result
"""

from .client import get_model_name, ping
from .judge import evaluate_with_judge
from .models import JudgeResult

__all__ = [
    "JudgeResult",
    "evaluate_with_judge",
    "get_model_name",
    "ping",
]
