"""Pydantic models returned by the LLM modules."""

from pydantic import BaseModel, Field


class JudgeResult(BaseModel):
    model_name: str
    judge_score: int = Field(ge=1, le=5)
    judge_feedback: str


# Where the 0-5 score cuts. 3 and not 4: on the 40-page run level 4 is never
# used, and moving the cut to 3 raises the correlation with f1 from +0.33 to
# +0.57. The boolean still decides the verdict; this is the comparison.
COHERENT_FROM = 3


class SelfCheckResult(BaseModel):
    """The model's verdict on an extraction, with no gold standard.

    ``coherent`` and ``coherence`` are the same question asked twice, as a
    yes/no and as a score. The boolean decides, because it is the wording that
    was measured; the score is kept beside it because the boolean is what
    flips between runs, and a score cannot flip the same way. Having both lets
    us pick the better one later with data.

    ``missing`` are quotes the model says did not reach the extraction. They
    are claims: only code looking for them can say which are real.
    """

    model_name: str
    coherent: bool
    coherence: int = Field(ge=0, le=5)
    # The passage the score is about, quoted from the Markdown or the HTML.
    # Empty when the model found nothing to object to.
    coherence_evidence: str = ""
    missing: list[str] = Field(default_factory=list)
    notes: str

    @property
    def coherent_by_score(self) -> bool:
        """What the verdict would be if the score decided. Only for comparison."""
        return self.coherence >= COHERENT_FROM
