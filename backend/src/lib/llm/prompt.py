"""Builds the prompt sent to the LLM judge.

Inputs are truncated to ``MAX_TEXT_CHARS`` to keep CPU inference fast.
"""

MAX_TEXT_CHARS = 1000


def build_judge_prompt(parsed_text: str, gold_text: str) -> str:
    """Return the prompt to send to the judge model."""
    parsed_truncated = parsed_text[:MAX_TEXT_CHARS]
    gold_truncated = gold_text[:MAX_TEXT_CHARS]
    return (
        f"""
        You are a judge of web parsing quality.

        Your only task is to COMPARE two texts: the PARSED DOCUMENT and the GOLD STANDARD.
        The GOLD STANDARD is the only source of truth. Measure how well the parsed document preserves its content.
        Both texts are written in Italian.

        Count something as a defect ONLY if you can verify it by comparing the two texts:
        - content present in the gold but missing from the parsed text;
        - content present in the parsed text but absent from the gold (added text or noise: menu, footer, banner, cookie notice, ads);
        - duplications in the parsed text;
        - structural problems (headings, sections, lists, tables, content order).

        MAIN RULE - do not invent defects:
        - Before writing that some content is "missing" (a law, an institution, a name, a date, a number, a reference...), CHECK that this content really appears in the gold standard.
        - If you cannot find it written in the gold standard below, it is NOT a defect: do not mention it.
        - Do not use outside knowledge: never add laws, institutions, facts or details that do not appear in the two texts.
        - If the two texts say the same thing with small formatting differences, it is NOT a defect.

        Use this scale strictly:
        - 1 = the two texts talk about different things, or almost all content is missing, or one of them is empty;
        - 2 = many errors: most of the content is lost, wrong, or buried under noise;
        - 3 = only partial content, with clear errors or significant noise;
        - 4 = good: the content matches almost entirely, only small imperfections remain;
        - 5 = nearly perfect extraction: no important part missing, no noise.

        Scoring rules:
        - First write the feedback listing the concrete defects you verified in the two texts, THEN assign a score consistent with those defects.
        - Penalize only defects you verified: do not lower the score for defects you cannot point to in the texts.
        - Give 4 or 5 ONLY if the informative content of the two texts really matches.
        - If the two texts are about different topics, give 1.

        Write the "feedback" field in Italian.

        Output examples:
        {{"feedback": "I due testi parlano di argomenti completamente diversi: nessun contenuto in comune.", "score": 1}}
        {{"feedback": "Manca circa metà del contenuto ed è presente molto rumore da menu e footer.", "score": 2}}
        {{"feedback": "Contenuto fedele al riferimento, solo piccole differenze di formattazione.", "score": 5}}

        Text extracted by the parser:
        {parsed_truncated}

        Reference text (Gold Standard):
        {gold_truncated}

        Answer ONLY with a JSON object in this format (the feedback must be one or two sentences at most, written in Italian):
        {{"feedback": "<difetti principali del testo>", "score": <intero tra 1 e 5>}}
        """
    )
