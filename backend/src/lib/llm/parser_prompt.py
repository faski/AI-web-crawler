"""Builds the prompts used for LLM-guided parsing.

Two prompts live here:

``build_parser_prompt``
    asks the model to turn raw HTML into the page's main content in Markdown.

``build_self_check_prompt``
    asks the model whether the text it produced is complete and coherent with
    respect to the HTML it came from, which is the reference-free check the
    project brief asks for.

Unlike the judge prompt, the HTML is **not** truncated here. Truncating the
input would silently change what the model is asked to parse, and the whole
point of this path is to measure what a model does with the page as it is; a
page that does not fit is reported as such by the caller instead.
"""

NOTHING_FOUND = "NOTHING_HERE"
"""What a fragment prompt answers when the fragment holds no content.

A model given a piece of nothing but minified JavaScript will otherwise
invent a plausible-looking paragraph rather than return an empty string,
so it is given a word to say instead, and the caller drops it.
"""

SELF_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "notes": {"type": "string"},
        "complete": {"type": "boolean"},
        "coherent": {"type": "boolean"},
    },
    "required": ["notes", "complete", "coherent"],
}


_EXTRACTION_RULES = """\
        MAIN CONTENT is what the page exists to show: the article and its
        title, or the data table the page is built around.

        REMOVE everything that surrounds it: navigation menus, sidebars,
        breadcrumbs, search boxes, login and newsletter forms, advertising,
        cookie and privacy banners, social sharing buttons, "related
        articles" and "most popular" lists, comments, and the footer.

        CHOOSING THE MAIN CONTENT. A page can show more than one candidate
        block: a landing page often embeds a promoted article, and an article
        often ends with a teaser for another one. Find the page's own subject
        first: read the <title> element and the FIRST <h1> in the document.
        The main content is the block that starts at that first <h1> and
        continues from there. Any other article-looking block - even one with
        its own heading, its own summary and many paragraphs, even one much
        longer than the real content - is promoted material: drop it. A block
        whose heading does not match the page <title> is not the main content.

        WHERE TO STOP. Encyclopedia and reference articles end with an
        apparatus that is not part of the content. As soon as you reach a
        heading named (in any language) Note, Notes, References, Riferimenti,
        Bibliografia, Bibliography, Fonti, Sources, Further reading, Voci
        correlate, See also, Altri progetti, Collegamenti esterni, External
        links, Controllo di autorita, Authority control, or a list of
        categories, STOP: output nothing from that heading onwards.

        Rules you must follow exactly:
        - Copy the text VERBATIM from the HTML. Do not paraphrase, do not
          summarise, do not shorten, do not fix what looks like a mistake.
        - Keep the original language of the page. Never translate.
        - Do not add anything that is not in the HTML: no titles you invented,
          no introductions, no comments of your own.
        - Remove footnote markers that appear inside the running text, such as
          [1], [23] or [N 1]. Keep the sentence, drop the marker.
        - For a link, keep only the text the reader sees, never the address.
        - Do not output images.
        - Keep the reading order of the page.
        - Use Markdown for the structure the page already has: '#' for the
          title and headings, '-' for lists, blank lines between paragraphs.
        - TABLES. Use a '|' table only when the table IS the content the page
          exists to show, like a list of results, prices or statistics. Do NOT
          reproduce the summary box that encyclopedia articles put beside the
          first paragraph (infobox, sinottico, fact sheet, "quick facts"): it
          is page furniture that repeats what the text already says. Leave it
          out entirely.
        - Never put raw HTML in the answer. No <br>, no <span>, no <a>, no
          attributes: if the HTML has markup inside a cell or a paragraph,
          keep the words and drop the tags.
        - Ignore the content of <script> and <style> elements: it is code, not
          text shown to the reader.
        - If the page is an article, the result starts with its title as a
          level-1 heading."""
"""What counts as the main content, and how it must be written out.

Shared by the whole-page prompt and the fragment prompt so that a page
read in one call and the same page read in four are judged against the
same target. Keeping two copies in step by hand is exactly how the two
paths would drift apart the next time these rules are tuned.
"""


def build_parser_prompt(url: str, html_text: str) -> str:
    """Return the prompt asking the model to extract the page as Markdown.

    The rules below state the same extraction target the hand-written parsers
    implement (see ``parsers/domains/wikipedia.py``): the same trailing
    sections are dropped, the same footnote markers are removed, links become
    their visible text. Giving the model a different target would make the
    comparison with Crawl4AI meaningless, since the two would be answering
    different questions.

    Args:
        url:       the source URL, which tells the model what kind of page
                   it is looking at.
        html_text: the raw HTML, passed through unchanged.
    """
    return (
        f"""
        You are a web page parser. You receive the raw HTML of a page and you
        return its main content as Markdown.

{_EXTRACTION_RULES}

        Answer with the Markdown ONLY. No preamble, no explanation, no code
        fences around the answer.

        Page URL: {url}

        Raw HTML:
        {html_text}

        Before answering, check these again - they are the ones most often
        missed on a long page:
        1. STOP at the first heading named Note, Notes, References,
           Riferimenti, Bibliografia, Fonti, Voci correlate, See also, Altri
           progetti, Collegamenti esterni, External links, Controllo di
           autorita, Authority control, or a category list. Nothing from there
           to the end of the page belongs in the answer.
        2. No infobox, no summary box, no fact sheet beside the first
           paragraph. A '|' table only if the table is what the page is about.
        3. No raw HTML tags or attributes anywhere in the answer.
        4. Copy the words verbatim; remove footnote markers like [1] or [23].

        Answer with the Markdown ONLY.
        """
    )


def build_chunk_prompt(
    url: str,
    page_title: str,
    page_heading: str,
    html_fragment: str,
    index: int,
    total: int,
) -> str:
    """Return the prompt for one fragment of a page too long to read at once.

    A fragment cannot be judged on its own: deciding what the main content is
    means knowing what the page is about, and that lives in the ``<title>``
    and the first ``<h1>``, which only one fragment contains. Both are passed
    in with every fragment so the rule the whole-page prompt states - the main
    content is the block that matches the page's subject - still means
    something here.

    The fragment-specific instructions come after the shared rules so they
    override them where the two disagree: a fragment must not open with the
    page title, and a fragment may legitimately contain nothing at all.

    Args:
        url:           the source URL of the whole page.
        page_title:    text of the page's <title>, extracted before the cut.
        page_heading:  text of the page's first <h1>, extracted before the cut.
        html_fragment: this piece of the raw HTML, passed through unchanged.
        index:         which fragment this is, counting from 1.
        total:         how many fragments the page was cut into.
    """
    return (
        f"""
        You are a web page parser. This page was too long to read in one go,
        so it was cut into {total} pieces and you are given piece {index}.
        You return the main content found IN THIS PIECE, as Markdown.

        THE PAGE THIS PIECE COMES FROM
        Page URL: {url}
        Page title: {page_title}
        Page first heading: {page_heading}

{_EXTRACTION_RULES}

        THIS IS A PIECE OF A PAGE, NOT A PAGE. Where the rules above assume a
        whole page, these win:
        - The piece may hold no readable content at all. Most of a raw page is
          <script> and <style>, and a piece can be nothing but code. When that
          is the case answer with exactly {NOTHING_FOUND} and nothing else.
        - Do not start with the page title. Write a level-1 heading only if
          the heading itself is inside this piece.
        - The piece may begin or end in the middle of a sentence, a list or a
          table. Copy what is there and stop. Do not finish the sentence, do
          not guess what came before or after, do not repeat something you
          think was in an earlier piece.
        - Judge every block against the page title and heading above. A block
          that belongs to another subject is promoted material even if this
          piece contains nothing else.
        - Say nothing about the piece itself: no numbering, no "this fragment
          contains", no summary of what you did.

        Piece {index} of {total} of the raw HTML:
        {html_fragment}

        Answer with the Markdown of this piece ONLY, or with exactly
        {NOTHING_FOUND} if it holds none of the page's main content.
        """
    )


def build_self_check_prompt(url: str, html_text: str, parsed_text: str) -> str:
    """Return the prompt asking the model to check its own extraction.

    The model gets the same HTML it parsed plus the Markdown it produced, and
    reports whether the extraction is complete (nothing important left behind)
    and coherent (nothing added, nothing out of order). No gold standard is
    involved.
    """
    return (
        f"""
        You check the quality of a web page extraction.

        You receive the RAW HTML of a page and the MARKDOWN that was extracted
        from it. Judge the Markdown against that HTML only. There is no
        reference text, and you must not use outside knowledge.

        WHAT COUNTS AS MAIN CONTENT. The extraction was asked to follow the
        rules below, and "complete" means complete according to THESE rules,
        not according to what you would have extracted yourself. Material the
        rules tell the extractor to leave out is not missing: it was dropped on
        purpose, and reporting it as missing would turn this check into a
        disagreement between two prompts instead of a reading of one page.

        Anything the rules order the extractor to drop is NEVER missing. Its
        absence is the extraction working. This covers, among others: images;
        link addresses; footnote markers such as [1] or [N 1]; infoboxes and
        summary boxes; and everything from a Note / References / Bibliografia /
        Voci correlate / Collegamenti esterni / See also / External links
        heading onwards. Do not report any of these as missing content, and do
        not treat their absence as a reason to answer false.

        What "complete" is really asking is narrower: is there a paragraph, a
        heading, a list item or a table row of the article's own body that a
        reader would expect to find and that is not there?

{_EXTRACTION_RULES}

        Answer two questions:

        - "complete": true if every part of the page's main content is present
          in the Markdown. False if something the reader would consider part of
          the article or of the main table is missing.
        - "coherent": true if the Markdown contains only content that really
          appears in the HTML, in the right order, without duplications and
          without leftovers from menus, banners, ads or the footer. False
          otherwise.

        Judge only what you can point to in the two inputs. If you cannot find
        a problem in the HTML or in the Markdown, there is no problem.

        In "notes", name the concrete problems you found, in one or two
        sentences. If you found none, say so.

        Page URL: {url}

        Raw HTML:
        {html_text}

        Extracted Markdown:
        {parsed_text}

        Answer ONLY with a JSON object in this exact format:
        {{"notes": "<problemi concreti trovati>", "complete": <true|false>, "coherent": <true|false>}}
        """
    )


def build_self_check_fragment_prompt(
    url: str,
    parsed_text: str,
    html_fragment: str,
    index: int,
    total: int,
) -> str:
    """Return the self-check prompt for one piece of an over-long page.

    The whole Markdown is shown against one piece of the HTML, so the two
    questions have to be narrowed to what a piece can actually answer.

    ``complete`` stays well posed: the model is asked whether the main content
    *of this piece* reached the Markdown, and a page is complete when every
    piece says so.

    ``coherent`` is narrowed to what this piece can refute: boilerplate it can
    see in its own HTML and also in the Markdown, or content of its own that
    the Markdown reproduces wrongly. What no piece can catch is invention -
    Markdown text that appears in no fragment at all - because a piece that
    does not contain a passage has no way to tell whether another piece did.
    That blind spot is a property of the check, not a bug in the prompt, and
    the whole-page check shares it whenever the invented text really is in the
    HTML, buried in a <script> the reader never sees.
    """
    return (
        f"""
        You check the quality of a web page extraction.

        The page was too long to read in one go, so its RAW HTML was cut into
        {total} pieces. You are given piece {index}, together with the whole
        MARKDOWN that was extracted from the complete page. There is no
        reference text, and you must not use outside knowledge.

        WHAT COUNTS AS MAIN CONTENT. The extraction was asked to follow the
        rules below, and "complete" means complete according to THESE rules,
        not according to what you would have extracted yourself. Material the
        rules tell the extractor to leave out is not missing: it was dropped on
        purpose, and reporting it as missing would turn this check into a
        disagreement between two prompts instead of a reading of one page.

        Anything the rules order the extractor to drop is NEVER missing. Its
        absence is the extraction working. This covers, among others: images;
        link addresses; footnote markers such as [1] or [N 1]; infoboxes and
        summary boxes; and everything from a Note / References / Bibliografia /
        Voci correlate / Collegamenti esterni / See also / External links
        heading onwards. Do not report any of these as missing content, and do
        not treat their absence as a reason to answer false.

        What "complete" is really asking is narrower: is there a paragraph, a
        heading, a list item or a table row of the article's own body that a
        reader would expect to find and that is not there?

{_EXTRACTION_RULES}

        THE MARKDOWN COVERS THE WHOLE PAGE, THIS HTML IS ONE PIECE OF IT.
        Most of the Markdown therefore comes from parts of the page you cannot
        see. That is normal and is never a problem. Judge only what this piece
        lets you judge:

        - "complete": true if every part of the page's MAIN CONTENT that is
          present IN THIS PIECE of HTML also appears in the Markdown. False if
          this piece contains a paragraph, a heading, a list or a table row
          that a reader would call part of the article or of the main table,
          and that is missing from the Markdown. If this piece holds no main
          content at all - many pieces are nothing but <script> and <style> -
          then nothing is missing and the answer is true.
        - "coherent": true unless THIS PIECE proves something wrong. It proves
          something wrong when text you can see in this piece is a navigation
          menu, a banner, an advertisement, a "related articles" list or a
          footer, and that same text appears in the Markdown; or when content
          of this piece is reproduced in the Markdown distorted, duplicated or
          out of order. Markdown text you simply cannot find in this piece is
          NOT a problem: it belongs to another piece. Do not report it.

        In "notes", name the concrete problems you found, in one or two
        sentences, quoting the text at issue. If you found none, say so.

        Page URL: {url}

        Piece {index} of {total} of the raw HTML:
        {html_fragment}

        Markdown extracted from the whole page:
        {parsed_text}

        Answer ONLY with a JSON object in this exact format:
        {{"notes": "<problemi concreti trovati>", "complete": <true|false>, "coherent": <true|false>}}
        """
    )
