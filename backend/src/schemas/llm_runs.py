"""Schemas for the LLM-parser run endpoints."""

from pydantic import BaseModel, Field


class LlmRunSummary(BaseModel):
    """One stored run with the numbers shown in the runs list."""

    id: int
    label: str
    model_name: str
    provider: str
    condition_name: str
    budget_tokens: int
    created_at: str
    ok: int
    too_long: int
    errors: int
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    excess_ratio: float | None = None
    seconds: float | None = None
    cpu_seconds: float | None = None
    input_tokens: int = 0
    skipped_tokens: int = 0
    # Model calls the run actually made, which equals the page count unless
    # chunking had to read some pages in several pieces.
    calls: int = 0
    chunked_pages: int = 0
    # Only a paid provider reports a cost; a local run leaves this empty.
    cost_eur: float | None = None
    eur_per_usd: float | None = None
    # True when the amount is the run's total read from the provider's
    # balance, with no per-page breakdown behind it.
    cost_is_aggregate: bool = False
    # Wall-clock seconds of the whole run, and what they cost on this
    # machine. It is the only cost a local run has. Not comparable with
    # cost_eur as it stands: an invoice includes a margin, this does not.
    wall_seconds: float | None = None
    local_cost_eur: float | None = None
    local_cost_electricity_eur: float | None = None
    local_cost_hardware_eur: float | None = None


class LlmDomainRow(BaseModel):
    """Averages of one run over one domain."""

    domain: str
    ok: int
    too_long: int
    errors: int
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    cosine: float | None = None
    excess_ratio: float | None = None
    seconds: float | None = None


class LlmPageRow(BaseModel):
    """One page of one run."""

    url: str
    domain: str
    status: str
    input_tokens: int | None = None
    seconds: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    excess_ratio: float | None = None
    fragments: int | None = None
    cost_eur: float | None = None
    # One dollar amount per model call, in the order the calls were made.
    call_costs_usd: list[float] | None = None
    # Share of the extraction found in the page's visible text. A low value
    # means the model described a page other than this one.
    grounded: float | None = None
    # What the model said about its own extraction, without the gold
    # standard. None means the run was never checked, which is not the same
    # as a page that was checked and passed.
    check_complete: bool | None = None
    check_coherent: bool | None = None
    check_notes: str | None = None
    check_cost_eur: float | None = None


class LlmSelfCheckSummary(BaseModel):
    """How a run's self-verdicts line up with the gold standard."""

    checked: int
    passed: int
    failed: int
    # Mean F1 on each side of the verdict. The distance between the two is
    # the measurement: a check that separates nothing leaves them level.
    f1_passed: float | None = None
    f1_failed: float | None = None
    f1_all: float | None = None
    not_complete: int = 0
    not_coherent: int = 0
    cost_eur: float | None = None
    # The same verdict as a 0-5 score, averaged over the pages that have one.
    # ``scored`` says how many they are.
    coherence_mean: float | None = None
    scored: int = 0
    # Omissions the code confirmed, over the whole run. What the model claimed
    # is in the per-page rows: the gap between the two is the point.
    omissions: int = 0
    # How the claimed omissions turned out, by verdict: confermata, presente,
    # inventata, troppo corta.
    quote_verdicts: dict[str, int] = Field(default_factory=dict)
    # How well each measure tracks the gold-standard F1. They are side by
    # side because one of them costs nothing.
    r_check: float | None = None
    r_grounded: float | None = None
    # The same verdict as a score: raw, and cut back to a yes/no at
    # ``coherent_from``. Kept beside r_check to compare the two.
    r_coherence: float | None = None
    r_coherence_cut: float | None = None
    coherent_from: int | None = None
    # The discarded half of the question, kept so the choice is visible.
    r_complete: float | None = None


class LlmQuoteCheck(BaseModel):
    """One passage the model said was missing, and where it really was.

    The verdict comes from the code, not from the model: only "confermata" is
    a real omission. The other three are different ways of getting the claim
    wrong, kept apart instead of collapsed into one boolean.
    """

    quote: str = ""
    verdict: str = ""
    in_page: bool = False
    in_extraction: bool = False


class LlmSelfCheckRow(BaseModel):
    """One checked page: what the gold standard says and what the model said."""

    url: str
    domain: str
    f1: float | None = None
    grounded: float | None = None
    check_complete: bool
    check_coherent: bool
    check_notes: str | None = None
    check_fragments: int | None = None
    check_cost_eur: float | None = None
    # The verdict as a 0-5 score, and the passage the model quoted for it.
    # None on an older run, checked before the score existed.
    check_coherence: int | None = None
    check_coherence_evidence: str | None = None
    # What the model said was missing, and what the code found. Claims without
    # omissions is the normal case, not a problem.
    check_claimed_missing: list[str] = Field(default_factory=list)
    check_quote_verdicts: list[LlmQuoteCheck] = Field(default_factory=list)
    check_omissions: int | None = None


class LlmComparisonRow(BaseModel):
    """One page seen in two runs at once."""

    url: str
    domain: str
    left_status: str | None = None
    right_status: str | None = None
    left_f1: float | None = None
    right_f1: float | None = None
    left_precision: float | None = None
    right_precision: float | None = None
    left_recall: float | None = None
    right_recall: float | None = None
    left_tokens: int | None = None
    right_tokens: int | None = None
    left_seconds: float | None = None
    right_seconds: float | None = None


class SaveRunRequest(BaseModel):
    """Payload used by the import script to store a finished run."""

    label: str
    model_name: str
    provider: str
    condition_name: str
    budget_tokens: int
    records: list[dict]


class LlmTextSide(BaseModel):
    """One run's output for one page."""

    run_id: int
    label: str
    model_name: str
    condition_name: str
    status: str
    f1: float | None = None
    precision: float | None = None
    recall: float | None = None
    excess_ratio: float | None = None
    seconds: float | None = None
    parsed_text: str = ""


class LlmTextComparison(BaseModel):
    """The text two runs produced for the same page."""

    url: str
    left: LlmTextSide | None = None
    right: LlmTextSide | None = None
