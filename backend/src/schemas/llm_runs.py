"""Schemas for the LLM-parser run endpoints."""

from pydantic import BaseModel


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
