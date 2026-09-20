"""Tables added after the initial schema, created on every startup.

``init.sql`` only runs the first time MariaDB initialises an empty data
directory, so a database that already exists would never see a new table.
These statements are idempotent and run at application startup instead, which
keeps an existing installation working without recreating it.
"""

from .connection import execute, fetch_one

# Metadata of one evaluation run: which model, through which provider, on
# which condition. ``label`` is unique so re-importing the same run updates it
# instead of piling up duplicates.
CREATE_LLM_RUNS = """
CREATE TABLE IF NOT EXISTS llm_runs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    label VARCHAR(191) CHARACTER SET ascii NOT NULL,
    model_name VARCHAR(255) CHARACTER SET ascii NOT NULL,
    provider VARCHAR(32) CHARACTER SET ascii NOT NULL,
    condition_name VARCHAR(64) NOT NULL,
    budget_tokens INT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_llm_run_label (label)
)
"""

# One row per page of a run. The primary key is (run_id, url), so url has to
# stay short enough for an index: 700 utf8mb4 characters are 2800 bytes, which
# with the INT comes to 2804, under InnoDB's 3072-byte limit. utf8mb4 and not
# ascii on purpose - the gold standard contains URLs with accented characters
# (Universita degli Studi...), and an ascii column silently replaces them with
# '?', which is what already happened to web_resources.url in the base project.
CREATE_LLM_PAGE_RESULTS = """
CREATE TABLE IF NOT EXISTS llm_page_results (
    run_id INT NOT NULL,
    url VARCHAR(700) CHARACTER SET utf8mb4 NOT NULL,
    domain VARCHAR(255) NOT NULL,
    status VARCHAR(16) CHARACTER SET ascii NOT NULL,
    input_tokens INT,
    html_kb INT,
    seconds DOUBLE,
    cpu_seconds DOUBLE,
    chars INT,
    precision_val DOUBLE,
    recall_val DOUBLE,
    f1 DOUBLE,
    cosine DOUBLE,
    jaccard DOUBLE,
    excess_ratio DOUBLE,
    extracted_count INT,
    sample_count INT,
    parsed_text LONGTEXT,
    note TEXT,
    PRIMARY KEY (run_id, url),
    FOREIGN KEY (run_id) REFERENCES llm_runs(id) ON DELETE CASCADE
)
"""


# Columns added after the table first shipped. Which of them are already
# there is read from information_schema before adding any: MariaDB has no
# portable "ADD COLUMN IF NOT EXISTS", and letting the duplicates fail instead
# meant the normal startup path ran a handful of failing statements, which is
# both noisy and how the connection leak in execute() stayed invisible.
LATER_COLUMNS = (
    "ALTER TABLE llm_page_results ADD COLUMN cpu_seconds DOUBLE",
    # How many pieces a page had to be cut into, and how many of them came
    # back empty. 1 and 0 for a page read in one call, which is every page of
    # every run made before chunking existed.
    "ALTER TABLE llm_page_results ADD COLUMN fragments INT",
    "ALTER TABLE llm_page_results ADD COLUMN empty_fragments INT",
    # What the page cost. The dollar amount is what the provider charged; the
    # euro amount is the conversion at the run's recorded rate. call_costs_usd
    # holds the per-call breakdown as a JSON array, because a page read in
    # four pieces was paid for four times and the provider reports only a
    # running total afterwards.
    "ALTER TABLE llm_page_results ADD COLUMN cost_usd DOUBLE",
    "ALTER TABLE llm_page_results ADD COLUMN cost_eur DOUBLE",
    "ALTER TABLE llm_page_results ADD COLUMN call_costs_usd TEXT",
    "ALTER TABLE llm_page_results ADD COLUMN prompt_tokens INT",
    "ALTER TABLE llm_page_results ADD COLUMN completion_tokens INT",
    # How much of the answer was found in the page's visible text. A
    # measurement only: nothing in the pipeline acts on it.
    "ALTER TABLE llm_page_results ADD COLUMN grounded DOUBLE",
    # The model's own verdict on the page it extracted: the brief's
    # reference-free check. Kept beside the gold-standard scores rather than
    # in a table of its own, because the only question worth asking of it is
    # whether it agrees with them. NULL on a run that was never checked, which
    # is not the same as a run that was checked and passed.
    "ALTER TABLE llm_page_results ADD COLUMN check_complete TINYINT(1)",
    "ALTER TABLE llm_page_results ADD COLUMN check_coherent TINYINT(1)",
    "ALTER TABLE llm_page_results ADD COLUMN check_notes TEXT",
    # How many calls the check itself took, and what it cost. Checking a page
    # is a second pass over the same HTML, so it has its own price.
    "ALTER TABLE llm_page_results ADD COLUMN check_fragments INT",
    "ALTER TABLE llm_page_results ADD COLUMN check_cost_eur DOUBLE",
    # Which provider served each call, and the id it gave the generation.
    # JSON arrays with one entry per call: a page read in four pieces can be
    # served by four different providers unless one is pinned, and without
    # this the run cannot say which.
    "ALTER TABLE llm_page_results ADD COLUMN providers TEXT",
    "ALTER TABLE llm_page_results ADD COLUMN generation_ids TEXT",
    # The 0-5 coherence score, beside the yes/no verdict. The two are the same
    # question asked twice: the boolean decides, the score is kept because it
    # is steadier across runs. evidence is the passage the score is about.
    "ALTER TABLE llm_page_results ADD COLUMN check_coherence TINYINT",
    "ALTER TABLE llm_page_results ADD COLUMN check_coherence_evidence TEXT",
    # The passages the model said were missing, and what the code found when
    # it looked for them: a JSON array of verdicts (confirmed, present,
    # invented, too short). A claim is not an omission until code says so.
    "ALTER TABLE llm_page_results ADD COLUMN check_claimed_missing TEXT",
    "ALTER TABLE llm_page_results ADD COLUMN check_quote_verdicts TEXT",
    "ALTER TABLE llm_page_results ADD COLUMN check_omissions INT",
    "ALTER TABLE llm_runs ADD COLUMN eur_per_usd DOUBLE",
    # Cost known for the run as a whole but not page by page. Runs made
    # before the per-call accounting existed can still report what they cost,
    # read from the provider's balance, without inventing a breakdown.
    "ALTER TABLE llm_runs ADD COLUMN measured_cost_eur DOUBLE",
)


def apply_schema() -> None:
    """Create the tables this version needs, and add any missing columns."""
    execute(CREATE_LLM_RUNS, ())
    execute(CREATE_LLM_PAGE_RESULTS, ())
    for statement in LATER_COLUMNS:
        table, column = _target_of(statement)
        if not _has_column(table, column):
            execute(statement, ())


def _target_of(statement: str) -> tuple[str, str]:
    """Return the table and column an ALTER ... ADD COLUMN statement targets."""
    words = statement.split()
    return words[2], words[5]


def _has_column(table: str, column: str) -> bool:
    """Return whether ``table`` already has ``column``."""
    return (
        fetch_one(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = DATABASE()
              AND table_name = ? AND column_name = ?
            """,
            (table, column),
        )
        is not None
    )
