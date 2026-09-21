"""Parser that hands the raw HTML to an LLM and asks for Markdown back.

The alternative to the Crawl4AI pipeline: instead of converting the HTML to
markdown and then stripping boilerplate with hand-written rules, the whole
page goes to a model that returns the main content directly.

The class does **not** implement ``ContentParser`` on purpose. That interface
takes markdown from Crawl4AI and returns clean text; this takes raw HTML and
returns Markdown. They are the two sides of the comparison, not two
implementations of the same thing.

A page too big for the budget is refused by default: it counts as
``too_long``, and the refusal is a measurement. With ``LLM_PARSER_CHUNKING``
on it is cut into pieces instead, each piece parsed on its own and the
answers joined - the brief's "manage context limits for very long HTML
pages". Off by default so a run made today still matches the runs made
before chunking existed.

Configuration is read from environment variables:
    LLM_PARSER_MAX_TOKENS      cap on the generated Markdown (default: 16384)
    LLM_PARSER_CONTEXT_TOKENS  input budget, in tokens (default: 128000)
    LLM_PARSER_CHUNKING        "1" to cut over-long pages instead of refusing
                               them (default: off)
    LLM_CHARS_PER_TOKEN        characters per token used to estimate the input
                               size (default: 3.0, close on average for HTML
                               but not a bound)
    LLM_TOKEN_SAFETY           factor the HTML budget is divided by, to cover
                               the estimate being wrong (default: 1.15). 1.0
                               reproduces the budget of the earlier runs.
"""

import json
import os
import re
from dataclasses import dataclass, replace

from ..evaluation import QuoteCheck, check_quotes, grounded_fraction
from ..llm import client
from ..llm.models import SelfCheckResult
from ..llm.parser_prompt import (
    NOTHING_FOUND,
    SELF_CHECK_SCHEMA,
    build_chunk_prompt,
    build_parser_prompt,
    build_self_check_fragment_prompt,
    build_self_check_prompt,
)
from . import html_chunker

DEFAULT_MAX_TOKENS = 16384
DEFAULT_CONTEXT_TOKENS = 128_000
DEFAULT_CHARS_PER_TOKEN = 3.0

# What the prompt costs in tokens, on top of the HTML it carries. Measured
# by sending each one and reading prompt_tokens back: 1084 the parser, 1127
# a fragment, 1804 and 1878 the two self-check prompts. 2500 and not 1900,
# so the next edit to a prompt has room before this has to move too.
PROMPT_OVERHEAD_TOKENS = 2500

# How wrong the token estimate may be, as a divisor of the HTML budget.
# estimate_tokens is not conservative: half the pages here tokenise worse
# than it says, the worst big one by 6.5%. Without this the budget has no
# margin at all - a page filled to the brim and 0.4% worse than estimated is
# already over the window, and the provider cuts it without a word.
DEFAULT_TOKEN_SAFETY = 1.15

# Room for the self-check answer. The reply is one small JSON object, but a
# model that reasons first spends tokens on that too. 1024 was not enough:
# one page in 260 quoted a long passage and the JSON was cut mid-string.
SELF_CHECK_ANSWER_TOKENS = 2048

# A fenced block wrapping the whole answer, which models add even when told
# not to. Only an outer fence is removed: see _clean_markdown.
OUTER_FENCE_PATTERN = re.compile(
    r"\A\s*```[a-zA-Z]*\s*\n(?P<body>.*)\n\s*```\s*\Z", re.DOTALL
)
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


# What the provider reports when it stopped because max_tokens ran out.
TRUNCATED = "length"


class TruncatedAnswerError(RuntimeError):
    """Raised when the model hit max_tokens and its answer is cut short.

    Half an extraction, not a bad one: scoring it would blame the model for a
    setting. Raised so the batch run records the page as an error and retries
    it, as it already does for a page that does not fit.
    """

    def __init__(self, url: str, max_tokens: int):
        super().__init__(
            f"the answer for {url} was cut at the {max_tokens} token ceiling: "
            "raise LLM_PARSER_MAX_TOKENS or the answer is missing its end"
        )
        self.url = url
        self.max_tokens = max_tokens


class HtmlTooLongError(RuntimeError):
    """Raised when the HTML does not fit the configured input budget.

    Carries both sizes so a run can report how many pages were out of reach,
    which is a result and not a crash.
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
    """Return an estimate of how many tokens ``text`` takes.

    A character ratio and not a real tokenizer, because the right tokenizer
    differs per model family and loading one means downloading it. Three
    characters per token is close on average for HTML but it is not a bound:
    against what the provider billed it is within 2% on average and out by up
    to 26% on a page full of inline JSON. A margin has to be added on top; see
    DEFAULT_TOKEN_SAFETY.
    """
    ratio = client.env_number("LLM_CHARS_PER_TOKEN", DEFAULT_CHARS_PER_TOKEN)
    return int(len(text) / ratio)


def _context_budget() -> int:
    """Return the input budget in tokens."""
    return client.env_number("LLM_PARSER_CONTEXT_TOKENS", DEFAULT_CONTEXT_TOKENS, int)


def _max_tokens() -> int:
    """Return the cap on the generated Markdown."""
    return client.env_number("LLM_PARSER_MAX_TOKENS", DEFAULT_MAX_TOKENS, int)


def _token_safety() -> float:
    """Return the factor the HTML budget is divided by.

    Configurable because it says how badly the estimate can miss on a given
    corpus, which is not a fact about this code. 1.0 gives back the budget the
    earlier runs used.
    """
    return client.env_number("LLM_TOKEN_SAFETY", DEFAULT_TOKEN_SAFETY)


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

    The context has to hold the prompt, the HTML and the answer together, so
    the HTML gets what is left over. Forgetting the answer in this sum is
    what makes a page look parsed while its last paragraphs were dropped.
    """
    budget = _context_budget() - PROMPT_OVERHEAD_TOKENS - _max_tokens()
    if budget <= 0:
        raise ValueError(
            f"no room left for HTML: a context of {_context_budget()} tokens "
            f"cannot hold a {PROMPT_OVERHEAD_TOKENS} token prompt and a "
            f"{_max_tokens()} token answer"
        )
    return int(budget / _token_safety())


def _self_check_budget(parsed_text: str) -> int:
    """Return how much HTML one self-check call may carry, in tokens.

    Three things where parsing had two: the prompt, the HTML and this time
    the Markdown as well, since the model has to compare them. The Markdown
    is charged in full even when the HTML is cut, because every piece is
    judged against the whole answer.
    """
    budget = (
        _context_budget()
        - PROMPT_OVERHEAD_TOKENS
        - SELF_CHECK_ANSWER_TOKENS
        - estimate_tokens(parsed_text)
    )
    if budget <= 0:
        raise ValueError(
            f"no room left for HTML: a context of {_context_budget()} tokens "
            f"cannot hold the prompt, a {estimate_tokens(parsed_text)} token "
            "answer to check, and the reply"
        )
    return int(budget / _token_safety())


def _clean_markdown(raw_response: str) -> str:
    """Strip a reasoning block and an outer code fence from the reply.

    Nothing else is removed. A model that opens with "Here is the extracted
    content:" is making a mistake that belongs in the measurements: cleaning
    it away would flatter the small models, the ones that do it.
    """
    cleaned = THINK_BLOCK_PATTERN.sub("", raw_response).strip()
    fenced = OUTER_FENCE_PATTERN.match(cleaned)
    if fenced:
        return fenced.group("body").strip()
    return cleaned


def _clean_fragment(raw_response: str) -> str:
    """Clean a fragment's answer, turning "nothing here" into an empty string.

    Compared stripped and case-folded: a model that answers ``nothing_here.``
    has still said there is nothing, and taking it as content would write the
    word into the page.
    """
    cleaned = _clean_markdown(raw_response)
    if cleaned.strip().strip(".").casefold() == NOTHING_FOUND.casefold():
        return ""
    return cleaned


def _without_repeated_title(text: str, context: html_chunker.PageContext) -> str:
    """Drop a level-1 heading repeating the page title at the start of ``text``.

    The fragment prompt says not to reopen the document with the page title,
    and a small model ignores it: an article spanning two fragments comes back
    with its ``# Titolo`` again in the middle. The seam repair cannot catch
    that, because it is not a repeat of the previous fragment's last lines.
    Doing it here does not depend on the model obeying.

    Only the first line, and only if it is an H1 carrying the page's own title
    or heading. An H1 that says something else is a real section heading.
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

    ``fragments`` is 1 for a page read in one call. More than that is what
    chunking cost on that page, which belongs on the "resources" axis of the
    comparison just like the token count.
    """

    text: str
    fragments: int
    empty_fragments: int
    # Share of the answer's long lines found in the page's visible text.
    # None when the answer has no line long enough to place.
    grounded: float | None = None
    # What the model replied before _clean_markdown touched it, one entry per
    # call. The stored Markdown is already cleaned, so without this there is
    # no way to see how often the cleaning fired, or what it removed.
    raw_responses: tuple[str, ...] = ()
    # One entry per call, in order, so a page read in four pieces shows what
    # each piece cost and not only the total.
    call_costs_usd: tuple[float, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Who served each call, in order. A tuple and not one name because a page
    # read in four pieces can be answered by four different providers: that is
    # how the routing was caught, and one name per page would have hidden it.
    providers: tuple[str, ...] = ()
    generation_ids: tuple[str, ...] = ()

    @property
    def cost_usd(self) -> float:
        """Return what the whole page cost, in dollars."""
        return sum(self.call_costs_usd)

    @property
    def cost_eur(self) -> float:
        """Return what the whole page cost, in euros at the configured rate."""
        return self.cost_usd * client.eur_per_usd()

    @property
    def providers_used(self) -> tuple[str, ...]:
        """Return the distinct providers that served this page, sorted.

        More than one name means the page was not read by one set of weights,
        so its numbers belong to no single model and must not be averaged with
        pages that were.
        """
        return tuple(sorted({p for p in self.providers if p}))


@dataclass(frozen=True)
class SelfCheckOutcome:
    """The model's verdict on an extraction, and what asking cost.

    ``fragments`` is 1 for a page checked in one call. A page too long is
    checked piece by piece, and ``per_fragment`` keeps the single verdicts so
    an overall "not complete" can be traced to the piece that said so.
    """

    result: SelfCheckResult
    fragments: int
    per_fragment: tuple[SelfCheckResult, ...] = ()
    # The quotes the model gave, after code looked for them. Empty if none.
    quote_checks: tuple[QuoteCheck, ...] = ()
    call_costs_usd: tuple[float, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Who served each call, in order. See ParseOutcome: a page read in pieces
    # can be answered by several providers unless one is pinned.
    providers: tuple[str, ...] = ()
    generation_ids: tuple[str, ...] = ()

    @property
    def omissions(self) -> tuple[QuoteCheck, ...]:
        """The claimed omissions that turned out to be real."""
        return tuple(c for c in self.quote_checks if c.is_omission)

    @property
    def complete(self) -> bool:
        """True if no quoted passage was really missing.

        The old boolean asked the model how it felt; this checks two strings.
        """
        return not self.omissions

    @property
    def cost_usd(self) -> float:
        """Return what checking this page cost, in dollars."""
        return sum(self.call_costs_usd)

    @property
    def cost_eur(self) -> float:
        """Return what checking this page cost, in euros at the configured rate."""
        return self.cost_usd * client.eur_per_usd()

    @property
    def providers_used(self) -> tuple[str, ...]:
        """Return the distinct providers that served this page, sorted.

        More than one name means the check was not done by one set of
        weights, so its numbers belong to no single model.
        """
        return tuple(sorted({p for p in self.providers if p}))


def _merge_checks(verdicts: list[SelfCheckResult], total: int) -> SelfCheckResult:
    """Combine per-fragment verdicts into one verdict for the page.

    The boolean is an AND and the score is the lowest any piece gave: only
    one piece saw the problem, so averaging would dilute a real fault away.

    Quotes are concatenated, since each piece quotes its own HTML, then cut to
    three because the check reports per page.

    Only the notes of the pieces that objected are kept, or the one that
    matters ends up buried under a dozen "nothing found".
    """
    worst = min(v.coherence for v in verdicts)
    agreed = all(v.coherent for v in verdicts)
    complaints = [
        f"[pezzo {index}/{total}] {verdict.notes.strip()}"
        for index, verdict in enumerate(verdicts, start=1)
        if not verdict.coherent or verdict.coherence < 5 or verdict.missing
    ]
    evidence = next(
        (v.coherence_evidence for v in verdicts
         if v.coherence == worst and v.coherence_evidence.strip()),
        "",
    )
    return SelfCheckResult(
        model_name=client.get_model_name(),
        coherent=agreed,
        coherence=worst,
        coherence_evidence=evidence,
        missing=[q for v in verdicts for q in v.missing][:3],
        notes=(
            " ".join(complaints)
            if complaints
            else f"Nessun problema trovato in {total} pezzi."
        ),
    )


class LlmParser:
    """Extracts a page's main content by prompting an LLM with its raw HTML."""

    def parse_html(self, url: str, html_text: str) -> str:
        """Return the page's main content as Markdown.

        The HTML is sent exactly as received, with nothing removed.

        Raises:
            HtmlTooLongError: if the HTML is over the budget and chunking is
                off. Sending it anyway lets the provider drop the overflow in
                silence, and the result looks like a parsing failure instead
                of a context one.
            TruncatedAnswerError: if the answer was cut at max_tokens. Same
                thing at the other end: half a page scored as a whole one
                reads as a bad model instead of a low ceiling.
        """
        return self.parse_html_detailed(url, html_text).text

    def parse_html_detailed(self, url: str, html_text: str) -> ParseOutcome:
        """Return the page's Markdown together with how it was read.

        With chunking off it behaves as every earlier run did: the page is
        checked against the context budget and read in one call, or refused.
        With chunking on it is measured against the room really left for HTML
        once prompt and answer are set aside, and cut only if it does not fit
        there.
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

        A reading, not an intervention. The pipeline is raw HTML in, Markdown
        out: a measurement that edited the answer would become another parser,
        which is the one thing this path must not hold.
        """
        return replace(outcome, grounded=grounded_fraction(outcome.text, html_text))

    def _parse_whole(self, url: str, html_text: str) -> ParseOutcome:
        """Read a page that fits, in a single call."""
        prompt = build_parser_prompt(url, html_text)
        raw_response, usage = client.generate_with_usage(
            prompt, max_tokens=_max_tokens()
        )
        if usage.finish_reason == TRUNCATED:
            raise TruncatedAnswerError(url, _max_tokens())
        return ParseOutcome(
            text=_clean_markdown(raw_response),
            fragments=1,
            empty_fragments=0,
            raw_responses=(raw_response,),
            call_costs_usd=(usage.cost_usd,),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            providers=(usage.provider,),
            generation_ids=(usage.generation_id,),
        )

    def _parse_in_fragments(self, url: str, html_text: str) -> ParseOutcome:
        """Cut the page, read each piece, and join the answers.

        The pieces are read one after the other. In parallel would be faster,
        but how much slower a page gets when it has to be read four times is
        one of the things being measured, and parallelism would hide it.
        """
        context = html_chunker.read_page_context(html_text)
        fragments = html_chunker.split_html(
            html_text, _fragment_budget(), estimate_tokens
        )

        answers, empty = [], 0
        raw_answers: list[str] = []
        costs: list[float] = []
        providers: list[str] = []
        generation_ids: list[str] = []
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
            # One cut piece is enough to spoil the page, so stop here instead
            # of stitching an answer with a hole in the middle.
            if usage.finish_reason == TRUNCATED:
                raise TruncatedAnswerError(url, _max_tokens())
            raw_answers.append(raw_response)
            costs.append(usage.cost_usd)
            providers.append(usage.provider)
            generation_ids.append(usage.generation_id)
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
            providers=tuple(providers),
            generation_ids=tuple(generation_ids),
        )

    def self_check(self, url: str, html_text: str, parsed_text: str) -> SelfCheckResult:
        """Ask the model whether its own extraction is complete and coherent."""
        return self.self_check_detailed(url, html_text, parsed_text).result

    def self_check_detailed(
        self, url: str, html_text: str, parsed_text: str
    ) -> SelfCheckOutcome:
        """Return the model's verdict on its own extraction, and what it cost.

        This is the reference-free check the brief asks for: the model sees the
        HTML and the Markdown, and no gold standard.

        A page that does not fit is cut with the same splitter parsing uses,
        so the check reaches the long pages too. Refusing them here would have
        left it measuring only the easy half of the corpus.
        """
        budget = _self_check_budget(parsed_text)
        if estimate_tokens(html_text) <= budget:
            outcome = self._check_whole(url, html_text, parsed_text)
        else:
            outcome = self._check_in_fragments(url, html_text, parsed_text, budget)
        # Against the whole page, not per fragment: a passage missing from
        # one piece may be in the next, and we would invent omissions.
        return replace(
            outcome,
            quote_checks=check_quotes(outcome.result.missing, parsed_text, html_text),
        )

    def _check_whole(
        self, url: str, html_text: str, parsed_text: str
    ) -> SelfCheckOutcome:
        """Check a page that fits, in a single call."""
        prompt = build_self_check_prompt(url, html_text, parsed_text)
        raw_response, usage = client.generate_with_usage(
            prompt,
            response_format=SELF_CHECK_SCHEMA,
            max_tokens=SELF_CHECK_ANSWER_TOKENS,
        )
        verdict = _parse_self_check(raw_response, usage.finish_reason)
        return SelfCheckOutcome(
            result=verdict,
            fragments=1,
            per_fragment=(verdict,),
            call_costs_usd=(usage.cost_usd,),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            providers=(usage.provider,),
            generation_ids=(usage.generation_id,),
        )

    def _check_in_fragments(
        self, url: str, html_text: str, parsed_text: str, budget: int
    ) -> SelfCheckOutcome:
        """Check a page piece by piece, judging each against the whole answer."""
        fragments = html_chunker.split_html(html_text, budget, estimate_tokens)

        verdicts: list[SelfCheckResult] = []
        costs: list[float] = []
        providers: list[str] = []
        generation_ids: list[str] = []
        prompt_tokens = completion_tokens = 0
        for number, fragment in enumerate(fragments, start=1):
            prompt = build_self_check_fragment_prompt(
                url=url,
                parsed_text=parsed_text,
                html_fragment=fragment,
                index=number,
                total=len(fragments),
            )
            raw_response, usage = client.generate_with_usage(
                prompt,
                response_format=SELF_CHECK_SCHEMA,
                max_tokens=SELF_CHECK_ANSWER_TOKENS,
            )
            verdicts.append(_parse_self_check(raw_response, usage.finish_reason))
            costs.append(usage.cost_usd)
            providers.append(usage.provider)
            generation_ids.append(usage.generation_id)
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens

        return SelfCheckOutcome(
            result=_merge_checks(verdicts, len(fragments)),
            fragments=len(fragments),
            per_fragment=tuple(verdicts),
            call_costs_usd=tuple(costs),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            providers=tuple(providers),
            generation_ids=tuple(generation_ids),
        )

    @staticmethod
    def _assert_fits(url: str, text: str) -> None:
        """Raise HtmlTooLongError if ``text`` is over the input budget."""
        estimated = estimate_tokens(text)
        budget = _context_budget()
        if estimated > budget:
            raise HtmlTooLongError(url, estimated, budget)


def _parse_self_check(raw_response: str, finish_reason: str = "") -> SelfCheckResult:
    """Parse the self-check reply, falling back to an explicit failure.

    ``finish_reason`` tells the two failures apart. A reply cut at the ceiling
    has no closing brace, so the search below finds nothing, and the old
    message blamed the model for the wrong format instead of naming the
    ceiling.
    """
    cleaned = THINK_BLOCK_PATTERN.sub("", raw_response)
    match = JSON_OBJECT_PATTERN.search(cleaned)
    if match is None:
        if finish_reason == TRUNCATED:
            return _self_check_fallback(
                f"Reply cut at the {SELF_CHECK_ANSWER_TOKENS} token ceiling, "
                f"not a formatting error: {raw_response[:200]}"
            )
        return _self_check_fallback(f"No JSON object in reply: {raw_response[:200]}")
    try:
        data = json.loads(match.group(0))
        # Capped here too: a model listing twenty quotes has stopped picking
        # the worst ones and is just listing what it sees.
        missing = [str(q) for q in (data.get("missing") or [])][:3]
        return SelfCheckResult(
            model_name=client.get_model_name(),
            coherent=bool(data["coherent"]),
            coherence=max(0, min(5, int(data["coherence"]))),
            coherence_evidence=str(data.get("coherence_evidence") or ""),
            missing=missing,
            notes=str(data["notes"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return _self_check_fallback(f"Invalid JSON from model: {match.group(0)[:200]}")


def _self_check_fallback(notes: str) -> SelfCheckResult:
    """Build a self-check result marking the verdict as unusable.

    Score 0 and coherent False, so a broken reply never counts as a pass, with
    the reason in ``notes``. No quotes: an unreadable answer is not evidence
    that anything is missing.
    """
    return SelfCheckResult(
        model_name=client.get_model_name(),
        coherent=False,
        coherence=0,
        coherence_evidence="",
        missing=[],
        notes=notes,
    )
