"""Parser that hands the raw HTML to an LLM and asks for Markdown back.

This is the alternative to the Crawl4AI pipeline: instead of converting the
HTML to markdown and then stripping boilerplate with hand-written rules, the
whole page is given to a model that returns the main content directly.

The class deliberately does **not** implement ``ContentParser``. That
interface takes markdown already produced by Crawl4AI and returns clean text;
this one takes raw HTML and returns Markdown. They are the two sides of the
comparison, not two implementations of the same contract.

A page that does not fit the input budget has two possible fates, chosen by
configuration. By default it is refused, which is what every run recorded so
far did: the page counts as ``too_long`` and the refusal is a measurement.
With ``LLM_PARSER_CHUNKING`` on, it is instead cut into pieces that do fit,
each piece is parsed on its own, and the answers are joined - the project
brief's "manage context limits for very long HTML pages". The default is off
so that a run made today still reproduces the runs made before chunking
existed; turning it on is a deliberate change of condition.

Configuration is read from environment variables:
    LLM_PARSER_MAX_TOKENS      cap on the generated Markdown (default: 16384)
    LLM_PARSER_CONTEXT_TOKENS  input budget, in tokens (default: 128000)
    LLM_PARSER_CHUNKING        "1" to cut over-long pages instead of refusing
                               them (default: off)
    LLM_CHARS_PER_TOKEN        characters per token used to estimate the input
                               size (default: 3.0, deliberately pessimistic
                               for HTML)
"""

import json
import os
import re
from dataclasses import dataclass, replace

from ..evaluation import grounded_fraction
from ..llm import client
from ..llm.models import SelfCheckResult
from ..llm.parser_prompt import (
    NOTHING_FOUND,
    SELF_CHECK_SCHEMA,
    build_chunk_prompt,
    build_parser_prompt,
    build_self_check_prompt,
)
from . import html_chunker

DEFAULT_MAX_TOKENS = 16384
DEFAULT_CONTEXT_TOKENS = 128_000
DEFAULT_CHARS_PER_TOKEN = 3.0

# What the prompt itself costs, in tokens, on top of the HTML it carries.
# Measured on the current prompt (about 4.4 KB) and rounded well up, because
# being wrong in this direction only wastes a little room, while being wrong
# in the other direction overflows the context and loses the tail in silence.
PROMPT_OVERHEAD_TOKENS = 2000

# A fenced block wrapping the whole answer, which models add even when told
# not to. Only an outer fence is removed: see _clean_markdown.
OUTER_FENCE_PATTERN = re.compile(
    r"\A\s*```[a-zA-Z]*\s*\n(?P<body>.*)\n\s*```\s*\Z", re.DOTALL
)
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class HtmlTooLongError(RuntimeError):
    """Raised when the HTML does not fit the configured input budget.

    Carries the two sizes so a batch run can report how many pages were out of
    reach for a given model, which is a result in itself rather than a crash.
    """

    def __init__(self, url: str, estimated_tokens: int, budget_tokens: int):
        super().__init__(
            f"HTML of {url} is about {estimated_tokens} tokens, "
            f"over the {budget_tokens} token budget"
        )
        self.url = url
        self.estimated_tokens = estimated_tokens
        self.budget_tokens = budget_tokens


def estimate_tokens(text: str) -> int:
    """Return a pessimistic estimate of how many tokens ``text`` takes.

    A character ratio is used rather than a real tokenizer because the right
    tokenizer differs per model family, and loading one would mean downloading
    it. The ratio errs low (3 chars per token) so the guard trips before the
    provider truncates, not after.
    """
    ratio = float(os.environ.get("LLM_CHARS_PER_TOKEN", DEFAULT_CHARS_PER_TOKEN))
    return int(len(text) / ratio)


def _context_budget() -> int:
    """Return the input budget in tokens."""
    return int(os.environ.get("LLM_PARSER_CONTEXT_TOKENS", DEFAULT_CONTEXT_TOKENS))


def _max_tokens() -> int:
    """Return the cap on the generated Markdown."""
    return int(os.environ.get("LLM_PARSER_MAX_TOKENS", DEFAULT_MAX_TOKENS))


def _chunking_enabled() -> bool:
    """Return whether over-long pages are cut instead of refused."""
    return os.environ.get("LLM_PARSER_CHUNKING", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _fragment_budget() -> int:
    """Return how much HTML one fragment may carry, in tokens.

    The context has to hold the prompt, the HTML and the answer at once, so
    the HTML gets what is left after the other two are set aside. Leaving the
    answer out of this sum is the mistake that makes a page look parsed while
    its last paragraphs were quietly dropped.
    """
    budget = _context_budget() - PROMPT_OVERHEAD_TOKENS - _max_tokens()
    if budget <= 0:
        raise ValueError(
            f"no room left for HTML: a context of {_context_budget()} tokens "
            f"cannot hold a {PROMPT_OVERHEAD_TOKENS} token prompt and a "
            f"{_max_tokens()} token answer"
        )
    return budget


def _clean_markdown(raw_response: str) -> str:
    """Strip a reasoning block and an outer code fence from the reply.

    Nothing else is removed. A model that prefixes its answer with "Here is the
    extracted content:" is making a mistake that belongs in the measurements:
    cleaning it away would flatter the smaller models, which are the ones that
    do it.
    """
    cleaned = THINK_BLOCK_PATTERN.sub("", raw_response).strip()
    fenced = OUTER_FENCE_PATTERN.match(cleaned)
    if fenced:
        return fenced.group("body").strip()
    return cleaned


def _clean_fragment(raw_response: str) -> str:
    """Clean a fragment's answer, turning "nothing here" into an empty string.

    The sentinel is compared on a stripped, case-folded line of its own: a
    model that answers ``nothing_here.`` has still said there is nothing, and
    treating that as content would put the word into the page.
    """
    cleaned = _clean_markdown(raw_response)
    if cleaned.strip().strip(".").casefold() == NOTHING_FOUND.casefold():
        return ""
    return cleaned


def _without_repeated_title(text: str, context: html_chunker.PageContext) -> str:
    """Drop a level-1 heading repeating the page title at the start of ``text``.

    The fragment prompt tells the model not to reopen the document with the
    page title, and a small model ignores it: a page whose article spans two
    fragments comes back with its ``# Titolo`` again in the middle, which the
    seam repair cannot catch because it is not a repetition of the previous
    fragment's last lines. Removing it here does not depend on the model
    obeying.

    Only the first line is examined, and only when it is an H1 whose text is
    the page's own title or heading. An H1 that says something else is a real
    section heading and stays.
    """
    lines = text.split("\n")
    if not lines or not lines[0].startswith("# "):
        return text

    heading = _fold_heading(lines[0][2:])
    known = {_fold_heading(context.title), _fold_heading(context.heading)}
    known.discard("")
    if heading not in known:
        return text
    return "\n".join(lines[1:]).strip()


def _fold_heading(text: str) -> str:
    """Return ``text`` with spacing collapsed and case folded, for comparison."""
    return " ".join(text.split()).strip().casefold()

@dataclass(frozen=True)
class ParseOutcome:
    """The Markdown of a page, and what it took to get it.

    ``fragments`` is 1 for a page read in one call. Anything higher is the
    extra cost chunking paid for that page, which belongs on the "resources"
    axis of the comparison as much as the token count does.
    """

    text: str
    fragments: int
    empty_fragments: int
    # Share of the answer's long lines found in the page's visible text.
    # None when the answer has no line long enough to place.
    grounded: float | None = None
    # What the model actually replied, before _clean_markdown touched it, one
    # entry per call. Kept so the cleaning can be audited: the stored Markdown
    # is post-cleaning, so without this there is no way to tell how often the
    # cleaning fired, or whether it ever removed something it should not have.
    raw_responses: tuple[str, ...] = ()
    # One entry per model call, in the order the calls were made, so a page
    # read in four pieces shows what each piece cost and not only the total.
    call_costs_usd: tuple[float, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        """Return what the whole page cost, in dollars."""
        return sum(self.call_costs_usd)

    @property
    def cost_eur(self) -> float:
        """Return what the whole page cost, in euros at the configured rate."""
        return self.cost_usd * client.eur_per_usd()


class LlmParser:
    """Extracts a page's main content by prompting an LLM with its raw HTML."""

    def parse_html(self, url: str, html_text: str) -> str:
        """Return the page's main content as Markdown.

        The HTML is sent exactly as received, with nothing removed.

        Raises:
            HtmlTooLongError: if the HTML exceeds the input budget and
                chunking is off. Sending it anyway would let the provider drop
                the overflow without saying so, and the result would look like
                a parsing failure instead of a context failure.
        """
        return self.parse_html_detailed(url, html_text).text

    def parse_html_detailed(self, url: str, html_text: str) -> ParseOutcome:
        """Return the page's Markdown together with how it was read.

        With chunking off the behaviour is the one every earlier run recorded:
        the page is checked against the context budget and read in a single
        call, or refused. With chunking on the page is measured against the
        room actually left for HTML once the prompt and the answer are set
        aside, and cut only if it does not fit there.
        """
        if not _chunking_enabled():
            self._assert_fits(url, html_text)
            return self._measured(html_text, self._parse_whole(url, html_text))

        if estimate_tokens(html_text) <= _fragment_budget():
            return self._measured(html_text, self._parse_whole(url, html_text))
        return self._parse_in_fragments(url, html_text)

    @staticmethod
    def _measured(html_text: str, outcome: ParseOutcome) -> ParseOutcome:
        """Record how much of the answer is on the page, and change nothing.

        This is a reading, not an intervention. The pipeline is raw HTML in,
        Markdown out; a measurement that edited the answer would stop being a
        measurement and become another parser, which is the one thing this
        path must not contain.
        """
        return replace(outcome, grounded=grounded_fraction(outcome.text, html_text))

    def _parse_whole(self, url: str, html_text: str) -> ParseOutcome:
        """Read a page that fits, in a single call."""
        prompt = build_parser_prompt(url, html_text)
        raw_response, usage = client.generate_with_usage(
            prompt, max_tokens=_max_tokens()
        )
        return ParseOutcome(
            text=_clean_markdown(raw_response),
            fragments=1,
            empty_fragments=0,
            raw_responses=(raw_response,),
            call_costs_usd=(usage.cost_usd,),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )

    def _parse_in_fragments(self, url: str, html_text: str) -> ParseOutcome:
        """Cut the page, read each piece, and join the answers.

        The pieces are read in order and in sequence. Reading them at the same
        time would be faster, but the cost of chunking - how much slower a page
        becomes when it has to be read four times - is one of the things the
        comparison is measuring, and hiding it behind parallelism would make
        the number meaningless.
        """
        context = html_chunker.read_page_context(html_text)
        fragments = html_chunker.split_html(
            html_text, _fragment_budget(), estimate_tokens
        )

        answers, empty = [], 0
        raw_answers: list[str] = []
        costs: list[float] = []
        prompt_tokens = completion_tokens = 0
        for number, fragment in enumerate(fragments, start=1):
            prompt = build_chunk_prompt(
                url=url,
                page_title=context.title,
                page_heading=context.heading,
                html_fragment=fragment,
                index=number,
                total=len(fragments),
            )
            raw_response, usage = client.generate_with_usage(
                prompt, max_tokens=_max_tokens()
            )
            raw_answers.append(raw_response)
            costs.append(usage.cost_usd)
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens

            answer = _clean_fragment(raw_response)
            if answers and any(answers):
                answer = _without_repeated_title(answer, context)
            if not answer:
                empty += 1
            answers.append(answer)

        text = html_chunker.stitch(answers)
        return ParseOutcome(
            text=text,
            fragments=len(fragments),
            empty_fragments=empty,
            call_costs_usd=tuple(costs),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            grounded=grounded_fraction(text, html_text),
            raw_responses=tuple(raw_answers),
        )

    def self_check(self, url: str, html_text: str, parsed_text: str) -> SelfCheckResult:
        """Ask the model whether its own extraction is complete and coherent.

        This is the reference-free check: the model sees the HTML and the
        Markdown, and no gold standard.

        Raises:
            HtmlTooLongError: if HTML and Markdown together exceed the budget.
        """
        self._assert_fits(url, html_text + parsed_text)
        prompt = build_self_check_prompt(url, html_text, parsed_text)
        raw_response = client.generate(
            prompt, response_format=SELF_CHECK_SCHEMA, max_tokens=1024
        )
        return _parse_self_check(raw_response)

    @staticmethod
    def _assert_fits(url: str, text: str) -> None:
        """Raise HtmlTooLongError if ``text`` is over the input budget."""
        estimated = estimate_tokens(text)
        budget = _context_budget()
        if estimated > budget:
            raise HtmlTooLongError(url, estimated, budget)


def _parse_self_check(raw_response: str) -> SelfCheckResult:
    """Parse the self-check reply, falling back to an explicit failure."""
    cleaned = THINK_BLOCK_PATTERN.sub("", raw_response)
    match = JSON_OBJECT_PATTERN.search(cleaned)
    if match is None:
        return _self_check_fallback(f"No JSON object in reply: {raw_response[:200]}")
    try:
        data = json.loads(match.group(0))
        return SelfCheckResult(
            model_name=client.get_model_name(),
            complete=bool(data["complete"]),
            coherent=bool(data["coherent"]),
            notes=str(data["notes"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return _self_check_fallback(f"Invalid JSON from model: {match.group(0)[:200]}")


def _self_check_fallback(notes: str) -> SelfCheckResult:
    """Build a self-check result marking the verdict as unusable.

    Both flags are False so a malformed reply is never counted as a pass; the
    reason is kept in ``notes`` so it stays visible in the results.
    """
    return SelfCheckResult(
        model_name=client.get_model_name(),
        complete=False,
        coherent=False,
        notes=notes,
    )
