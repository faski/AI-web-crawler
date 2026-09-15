"""Splits an HTML page into fragments small enough for a model's context.

Some pages do not fit in any context window we can afford: the CNBC articles
in this project run from 458k to 561k estimated tokens, against a 262k budget.
Sending them anyway is not an option, because a provider that overflows drops
the tail silently and the result looks like a parsing failure rather than a
context failure. This module is the other answer: cut the page into pieces,
parse each piece, and put the answers back together.

Two properties matter for the cut.

**It follows the markup.** Cutting at a character offset would leave a tag
open at the end of one fragment and orphaned at the start of the next, and a
model handed a fragment beginning mid-attribute has to guess what it is
looking at. Here the page is parsed and the cut falls between elements.

**It descends.** The obvious cut - one fragment per top-level element - does
nothing on a real page: a CNBC ``<body>`` has a single child wrapping the
whole document. So an element that is itself too large is opened and its
children are considered in turn, as deep as needed.

**It keeps everything.** The whole document is divided, ``<head>`` included,
and the fragments concatenate back to the parsed page. It would be cheaper to
throw the head away - on a CNBC page it is 830 KB of one inline ``<style>``,
and 94% of the whole page is CSS and script that no reader ever sees - but
dropping it here would quietly turn the raw-HTML condition into a pruned one,
and pruning is the other experiment. Deciding what is worth keeping is the
model's job; this module only makes the page small enough to ask.

The module deliberately knows nothing about prompts or LLM clients: it turns
HTML into a list of HTML strings, and is used by ``llm_parser``.
"""

import re
from dataclasses import dataclass
from typing import Callable, Iterator

from bs4 import BeautifulSoup, NavigableString, Tag

# How many lines at a seam are compared when looking for a repeated block.
# Two fragments that overlap usually repeat a heading and a paragraph or two;
# beyond this window the search costs more than it recovers.
SEAM_WINDOW_LINES = 12

WHITESPACE_RUN = re.compile(r"\s+")


@dataclass(frozen=True)
class PageContext:
    """What a fragment needs to know about the page it was cut from.

    A fragment on its own cannot tell main content from promoted material:
    that judgement needs the page's subject, which is exactly what the parser
    prompt tells the model to read from ``<title>`` and the first ``<h1>``.
    Cutting the page would throw that away for every fragment but one, so it
    is extracted once and passed to all of them.
    """

    title: str
    heading: str


def read_page_context(html_text: str) -> PageContext:
    """Return the page's ``<title>`` and first ``<h1>``, as plain text."""
    soup = BeautifulSoup(html_text, "lxml")
    return PageContext(
        title=_visible_text(soup.title),
        heading=_visible_text(soup.h1),
    )


def split_html(
    html_text: str,
    budget_tokens: int,
    measure: Callable[[str], int],
) -> list[str]:
    """Cut ``html_text`` into fragments of at most ``budget_tokens`` each.

    Args:
        html_text:     the raw HTML of the page.
        budget_tokens: the size each fragment must stay under. This is the
                       room left for the HTML alone, so the caller must
                       subtract what the prompt itself takes.
        measure:       returns the token cost of a string. Passed in rather
                       than imported so this module stays independent of how
                       the parser estimates tokens.

    Returns:
        The fragments, in reading order. Concatenating them reproduces the
        parsed page exactly, so nothing is dropped by the cut. A page that
        already fits comes back as a single fragment, so the caller can always
        treat the result the same way.
    """
    if budget_tokens <= 0:
        raise ValueError(f"budget_tokens must be positive, got {budget_tokens}")
    if measure(html_text) <= budget_tokens:
        return [html_text]

    soup = BeautifulSoup(html_text, "lxml")
    blocks = list(_as_blocks(soup, budget_tokens, measure))
    return _pack(_refined(soup, blocks, budget_tokens, measure), budget_tokens, measure)


def _refined(
    soup: BeautifulSoup,
    blocks: list[str],
    budget_tokens: int,
    measure: Callable[[str], int],
) -> list[str]:
    """Re-cut the page into blocks small enough to be spread evenly.

    ``_as_blocks`` stops opening an element as soon as it fits the budget, so
    a single 237.616-token ``<style>`` comes back whole and fills a fragment
    on its own. No amount of clever packing can even that out: the packer can
    only move blocks around, and one block is already the size of a fragment.

    So once the fragment count is known, the page is walked again with the
    even share as the limit. The walk goes deeper, the blocks come out
    smaller, and the packer has something to work with. Nothing is discarded
    on the second pass either - the finer blocks still add up to the page.
    """
    sizes = [measure(block) + 1 for block in blocks]
    fewest = len(_group(sizes, budget_tokens))
    if fewest <= 1:
        return blocks

    share = -(-sum(sizes) // fewest)
    finer = list(_as_blocks(soup, share, measure))
    # Only worth it if the finer cut still needs no extra model call. It
    # normally needs fewer or the same, but a pathological page could pack
    # worse, and one more call is a price this is not allowed to charge.
    if len(_group([measure(block) + 1 for block in finer], budget_tokens)) > fewest:
        return blocks
    return finer


def stitch(fragment_texts: list[str]) -> str:
    """Join the Markdown answers of consecutive fragments into one document.

    Empty answers are dropped: on raw HTML most fragments hold nothing but
    scripts, and the prompt allows the model to say so. Where the end of one
    answer repeats the start of the next - the usual effect of an element that
    straddles a cut - the repetition is removed once.
    """
    joined: list[str] = []
    for text in fragment_texts:
        cleaned = text.strip()
        if not cleaned:
            continue
        if not joined:
            joined.append(cleaned)
            continue
        joined.append(_drop_seam_overlap(joined[-1], cleaned))
    return "\n\n".join(part for part in joined if part).strip()


def _as_blocks(
    node: Tag,
    budget_tokens: int,
    measure: Callable[[str], int],
) -> Iterator[str]:
    """Yield the HTML of ``node``'s children, opening any child that is too big.

    A child that fits is yielded whole. A child that does not is opened and
    its own children are considered the same way, which is what makes the
    single-wrapper page splittable. A child that is too big and has no
    children left to open - a very long script, or one enormous run of text -
    is cut by length as a last resort, since there is no markup to cut along.

    Text nodes are emitted through ``output_ready`` rather than ``str``, which
    would drop the delimiters of a doctype or a comment and leave the
    fragments no longer adding up to the page. Whitespace-only nodes are kept
    for the same reason: they cost nothing and they keep the invariant exact.
    """
    for child in node.children:
        if isinstance(child, NavigableString):
            yield from _cut_by_length(
                child.output_ready(formatter=None), budget_tokens, measure
            )
            continue
        if not isinstance(child, Tag):
            continue

        markup = str(child)
        if measure(markup) <= budget_tokens:
            yield markup
        elif any(isinstance(g, Tag) for g in child.children):
            opening, closing = _tag_delimiters(child)
            yield opening
            yield from _as_blocks(child, budget_tokens, measure)
            yield closing
        else:
            yield from _cut_by_length(markup, budget_tokens, measure)


def _tag_delimiters(tag: Tag) -> tuple[str, str]:
    """Return the opening and closing markup of ``tag``, without its contents.

    Opening a tag to reach its children would otherwise lose the tag itself,
    and the fragments would no longer add up to the page. There is no public
    call for this in BeautifulSoup, so the serialiser's own helper is used;
    the version is pinned in requirements.txt, and ``verify_chunker.py``
    fails loudly if an upgrade ever changes what it returns.

    The formatter has to be the one ``str(tag)`` uses, "minimal", and not a
    freshly built ``HTMLFormatter``: the two escape attributes differently, so
    a default-built formatter writes ``class="[&>x]"`` where the page says
    ``class="[&amp;&gt;x]"`` and the fragments stop matching by a few
    characters on pages that use such class names.
    """
    formatter = tag.formatter_for_name("minimal")
    return (
        tag._format_tag(None, formatter, opening=True),
        tag._format_tag(None, formatter, opening=False),
    )


def _cut_by_length(
    text: str,
    budget_tokens: int,
    measure: Callable[[str], int],
) -> Iterator[str]:
    """Cut a string with no usable markup into pieces under the budget.

    The step starts from the budget's share of the string and is then shrunk
    until a piece really measures under budget. The first estimate assumes
    ``measure`` is proportional to length, which it is not exactly - it rounds
    down - so a step taken on trust can come out slightly too long.
    """
    total = measure(text)
    if total <= budget_tokens:
        yield text
        return

    step = max(1, len(text) * budget_tokens // total)
    while step > 1:
        measured = measure(text[:step])
        if measured <= budget_tokens:
            break
        step = max(1, step * budget_tokens // measured - 1)

    for start in range(0, len(text), step):
        yield text[start : start + step]


def _pack(
    blocks: list[str],
    budget_tokens: int,
    measure: Callable[[str], int],
) -> list[str]:
    """Group consecutive blocks into fragments of roughly equal size.

    Filling each fragment to the brim before opening the next is the obvious
    way to use the fewest calls, and it produced 780 / 237.616 / 221.091 /
    101.994 tokens on a CNBC page: one fragment nearly empty, the others full.
    That shape is worse than it looks, because the more tightly a fragment is
    packed the more likely the article is to be cut somewhere inside it, and a
    cut inside the article is where content gets lost or a heading gets
    written twice.

    So the fragment count is decided first - the fewest the budget allows -
    and the blocks are then spread over that many fragments as evenly as the
    block sizes permit. The number of model calls is unchanged; only the
    shape of the pieces is.

    Reading order is preserved: blocks are never reordered to fill a fragment
    more tightly, because a fragment is going to be read as a piece of the
    page and its parts must still follow one another.

    Each block is charged one token more than it measures. ``measure`` rounds
    down, so the sum of the parts can fall below the measure of the whole by
    almost a token per block - and a fragment built from hundreds of blocks
    was landing over budget while the running total said it still fitted.
    """
    sizes = [measure(block) + 1 for block in blocks]
    fewest = len(_group(sizes, budget_tokens))
    limit = _evenest_limit(sizes, budget_tokens, fewest)
    return ["".join(blocks[i] for i in group) for group in _group(sizes, limit)]


def _group(sizes: list[int], limit: int) -> list[list[int]]:
    """Return the block indices grouped into runs that each stay under ``limit``.

    A single block larger than ``limit`` becomes a group of its own rather
    than being dropped or split again: by the time packing runs, every block
    is already known to fit the budget, and ``limit`` here is only a target.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    current_size = 0

    for index, size in enumerate(sizes):
        if current and current_size + size > limit:
            groups.append(current)
            current, current_size = [], 0
        current.append(index)
        current_size += size

    if current:
        groups.append(current)
    return groups


def _evenest_limit(sizes: list[int], budget_tokens: int, fewest: int) -> int:
    """Return the smallest per-fragment limit that still needs ``fewest`` groups.

    Lowering the limit spreads the blocks out; lowering it too far needs one
    more fragment, which costs one more model call. The smallest limit that
    does not is the evenest split available without paying for it, and it is
    found by binary search between the perfectly even share and the budget.
    """
    if fewest <= 1:
        return budget_tokens

    low, high = -(-sum(sizes) // fewest), budget_tokens
    while low < high:
        middle = (low + high) // 2
        if len(_group(sizes, middle)) <= fewest:
            high = middle
        else:
            low = middle + 1
    return low


def _drop_seam_overlap(previous: str, current: str) -> str:
    """Return ``current`` without the opening lines that end ``previous``.

    The comparison ignores spacing and case so that a heading re-emitted with
    different wrapping still counts as the same line.
    """
    tail = _content_lines(previous)[-SEAM_WINDOW_LINES:]
    head_lines = current.split("\n")
    head = _normalise_all(head_lines)

    for length in range(min(len(tail), len(head)), 0, -1):
        if tail[-length:] == head[:length]:
            return "\n".join(head_lines[length:]).strip()
    return current


def _content_lines(text: str) -> list[str]:
    """Return the normalised lines of ``text``, blank lines included.

    Blank lines are kept so that an overlap found here lines up with the same
    stretch of lines in the other fragment.
    """
    return _normalise_all(text.split("\n"))


def _normalise_all(lines: list[str]) -> list[str]:
    """Return each line with runs of whitespace collapsed and case folded."""
    return [WHITESPACE_RUN.sub(" ", line).strip().casefold() for line in lines]


def _visible_text(node: Tag | None) -> str:
    """Return the text of ``node`` on one line, or "" if it is absent."""
    if node is None:
        return ""
    return WHITESPACE_RUN.sub(" ", node.get_text(" ", strip=True)).strip()
