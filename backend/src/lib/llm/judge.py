"""Evaluate parser output using the LLM judge.

Returns a ``JudgeResult`` with the model name, an integer score (1-5) and
a short feedback. If the model reply cannot be parsed as JSON, a fallback
result with score 1 and a diagnostic feedback is returned.
"""

import json
import re

from ..evaluation.tokens import strip_markdown
from . import client
from .models import JudgeResult
from .prompt import build_judge_prompt


THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

# Forces the model to answer with a valid JSON object in this exact shape.
# "feedback" is listed BEFORE "score" on purpose: the model generates the
# fields in this order, so it writes its analysis first and only then picks a
# score consistent with it (instead of anchoring to a default value).
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "feedback": {"type": "string"},
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["feedback", "score"],
}


def evaluate_with_judge(parsed_text: str, gold_text: str) -> JudgeResult:
    """Ask the LLM to score the parsed text against the gold text.

    Markdown is stripped from both inputs so the judge compares content only.
    """
    prompt = build_judge_prompt(strip_markdown(parsed_text), strip_markdown(gold_text))
    raw_response = client.generate(prompt, response_format=JUDGE_SCHEMA)
    return _parse_judge_response(raw_response)


def _extract_json_object(raw_response: str) -> str | None:
    """Return the first ``{...}`` block in the response, or None."""
    cleaned = THINK_BLOCK_PATTERN.sub("", raw_response)
    match = JSON_OBJECT_PATTERN.search(cleaned)
    return match.group(0) if match else None


def _parse_judge_response(raw_response: str) -> JudgeResult:
    """Parse the model output into a JudgeResult, falling back on errors."""
    json_text = _extract_json_object(raw_response)
    if json_text is None:
        return _fallback_result(f"No JSON object found in response: {raw_response[:200]}")

    try:
        data = json.loads(json_text)
        score = int(data["score"])
        feedback = str(data["feedback"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return _fallback_result(f"Invalid JSON from judge: {json_text[:200]}")

    score = max(1, min(5, score))
    return JudgeResult(
        model_name=client.get_model_name(),
        judge_score=score,
        judge_feedback=feedback,
    )


def _fallback_result(feedback: str) -> JudgeResult:
    """Build a default JudgeResult with score 1 and a diagnostic feedback."""
    return JudgeResult(
        model_name=client.get_model_name(),
        judge_score=1,
        judge_feedback=feedback,
    )
