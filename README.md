# Creeping Crawler

A Python tool that evaluates content extraction quality from web pages. It crawls URLs with Crawl4AI, parses the resulting markdown with URL-specific parsers, and scores extraction quality against gold standard samples using two evaluation groups: token-level (precision/recall/F1) and similarity (cosine/jaccard/excess ratio).

## Requirements

- [Conda](https://docs.conda.io/en/latest/)
- Python 3.11

## Setup

```bash
make envs
```

This creates two conda environments (`creeping-crawler-backend` and `creeping-crawler-frontend`) and installs all dependencies.

## Configuration

Everything has a working default, so `make up` runs the whole stack with no
configuration at all. Settings live in `.env`, which is not in the repository
because it holds an API key:

```bash
cp .env.example .env
```

`.env.example` documents every variable. Two of them decide what the LLM
parser can do:

| Variable | Default | What it changes |
|---|---|---|
| `PARSE_WITH_LLM` | `0` | Lets the `/parser` page call a model for a **new** extraction. Off, because such a call either costs money or takes minutes. |
| `LLM_PROVIDER` | `ollama` | Who answers: the local model, or `openrouter` with `OPENROUTER_API_KEY`. |

Without any of this the LLM parser still works on the forty gold standard
pages. The first boot seeds the database from two tracked directories - the
pages from `gs_data/` and the finished runs from `runs_data/` - so `/parser`
can show the Markdown the model really produced and `/llm-runs` has something
to compare. Only a **new** extraction needs the two settings above.

Both are seeded once and never again: a run imported later is not overwritten
on the next boot.

A warning about the local model. It runs on CPU inside Docker, with no GPU,
at roughly 25 tokens per second: the smallest page of the gold standard is
75,000 tokens, so reading it alone would take about 50 minutes and in
practice the request times out. The measurements in this project were made
against a remote provider for that reason.

## Running

**With Docker Compose (recommended):**

```bash
make up                  # build + start all services in background
make logs                # follow logs of all services (Ctrl+C to exit)
make down                # stop and remove all services
```

Each target accepts an optional service name as positional argument to target a single container (`mariadb`, `ollama`, `backend`, `frontend`):

```bash
make up backend          # start only backend (and its dependencies)
make logs ollama         # follow logs of ollama
make down mariadb        # stop and remove only mariadb
```

**Without Docker (two terminals):**

```bash
make run-backend    # http://localhost:8003
make run-frontend   # http://localhost:8004
```

## Development

```bash
make freeze         # Snapshot requirements.txt files from current envs
make delete-envs    # Remove conda environments
```

## Project Structure

```
backend/
└── src/
    ├── server.py                 # FastAPI app entry point
    ├── routes/                   # API route handlers
    │   ├── domains.py
    │   ├── parse.py
    │   ├── evaluate.py
    │   └── gold.py
    ├── schemas/                  # Pydantic request/response models
    │   ├── domains.py            # DomainsResponse
    │   ├── parse.py              # ParseRequest, ParseResponse
    │   ├── evaluate.py           # EvaluateRequest, EvaluateResponse, TokenLevelEval, SimilarityEval
    │   └── gold.py               # GoldStandardResponse, FullGoldStandardResponse, GoldStandardUrlsResponse
    └── lib/                      # Core library (no FastAPI dependencies except utils)
        ├── utils.py              # domain_of, assert_supported_domain
        ├── crawling/
        │   ├── crawler.py        # fetch_page, fetch_page_from_html, fetch_page_for_url
        │   └── domains/          # Domain-specific CrawlerRunConfig
        │       ├── registry.py
        │       ├── wikipedia.py
        │       ├── espn.py
        │       ├── cnbc.py
        │       └── xe.py
        ├── parsers/
        │   ├── base.py           # ContentParser abstract class
        │   ├── default.py        # PassThroughParser (fallback)
        │   ├── registry.py       # URL → parser lookup
        │   └── domains/          # Domain-specific parser implementations
        │       ├── wikipedia.py
        │       ├── espn.py
        │       ├── cnbc.py
        │       └── xe.py
        ├── evaluation/
        │   ├── tokens.py         # strip_markdown, extract_unique_tokens
        │   ├── token_level.py    # Set-based metrics: precision, recall, F1
        │   └── similarity.py     # Vector-based metrics: cosine, Jaccard, excess ratio
        └── gold_standard/
            ├── urls.py           # Domain/URL listing from gs_data/
            └── gold.py           # Gold text lookup

frontend/
├── static/
│   └── style.css
├── templates/                    # Jinja2 HTML templates
│   ├── base.html.jinja
│   ├── index.html.jinja          # Home page (URL selector)
│   ├── result.html.jinja         # Results page (panels + Monaco diff)
│   └── error.html.jinja
└── src/
    ├── app.py                    # FastAPI app entry point
    ├── client.py                 # HTTP client for backend API calls
    ├── templates.py              # Jinja2 template engine setup
    ├── utils.py                  # strip_markdown, helpers
    └── routes/
        ├── index.py              # GET /
        └── compare.py            # GET /compare?url=
```

## REST API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/parse` | Parse `{url, local?, parser?, fresh?}` — `parser` is `crawl4ai` or `llm`, `fresh` asks the model again instead of reading a stored extraction |
| GET | `/domains` | List supported domains |
| GET | `/gold_standard?url=` | Gold standard entry for a URL |
| GET | `/gold_standard_urls?domain=` | GS URLs for a domain |
| POST | `/evaluate` | Score `{parsed_text, gold_text}` (quantitative metrics) |
| POST | `/evaluate_judge` | Score `{parsed_text, gold_text}` via LLM judge |
| GET | `/full_gs_eval?domain=` | Averaged scores (incl. judge) across a domain's GS |
| POST | `/add_web_resource` | Insert `{url, html_text}` into the DB |
| POST | `/add_gold_standard` | Insert `{url, gold_text}` into the DB |
| DELETE | `/web_resource` | Remove `{url}` from web_resources (cascades) |
| DELETE | `/gold_standard` | Remove `{url}` from gold_standard only |
| GET | `/db_stats` | Per-domain counts and average evaluations |
| GET | `/db_schema` | JSON description of the DB tables |
| GET | `/status` | Health of backend / database / ollama, plus whether the LLM parser is allowed and who would answer |
| GET | `/llm_runs` | Stored LLM-parser runs with their headline numbers |
| GET | `/llm_runs/{id}/pages` | One run page by page; `/domains` and `/self_check` give the same run by domain and its self-check |
| GET | `/llm_runs/compare?left=&right=` | Two runs side by side; `/compare_text` adds the text of one page |

Errors: `400` unsupported domain · `403` LLM parser off · `404` URL not in DB · `413` HTML over the context budget · `502` unreachable URL or unusable model answer.

## LLM-guided parsing

Besides the Crawl4AI pipeline the project includes a second parser: the raw
HTML goes to a language model, which returns the page's main content as
Markdown. The two can be compared on the same pages and against the same gold
standard.

- `/parser` runs either pipeline on a single URL and scores the result. For a
  gold standard URL the LLM side shows the extraction stored by a finished
  run, which costs nothing; `PARSE_WITH_LLM=1` is only needed to ask for a
  new one.
- `/llm-runs` compares whole runs: quality, speed and cost, page by page,
  plus the model's own check of its extractions.

A run is produced from the command line and then imported:

```bash
python scripts/run_llm_eval.py --out output/run.json --chunk --openrouter
python scripts/run_self_check.py --run output/run.json \
    --out output/selfcheck.json --openrouter
python scripts/import_llm_run.py output/run.json --label "Qwen 3.5 9B"
python scripts/import_self_check.py output/selfcheck.json \
    --run-label "Qwen 3.5 9B"
```

`--openrouter` is required every time: without it the scripts use the local
model, so no run can spend money by accident.

A run that should ship with the project is then written to `runs_data/`,
which the next clone loads on first boot:

```bash
python scripts/export_run_seed.py --all
```

The seed files hold what the pages show - the extracted Markdown, the scores,
the self-check verdicts - and about 400 KB each. The raw HTML is not in them:
it already travels in `gs_data/`.

## Supported domains

Gold standard data lives in `gs_data/`. A domain is supported when it has a corresponding `<domain>_gs.json` file there. Currently supported: Italian Wikipedia, ESPN, CNBC, XE.

## Metrics

Both the parsed text and the gold standard are stripped of markdown before scoring.

### Token Level Eval — set-based

Unique token sets (whitespace splitting after markdown stripping):

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision** | `\|parsed ∩ gold\| / \|parsed\|` | How much of the extracted content is relevant |
| **Recall** | `\|parsed ∩ gold\| / \|gold\|` | How much of the gold content was extracted |
| **F1** | `2 · P · R / (P + R)` | Harmonic mean of precision and recall |

### Similarity Eval — frequency-vector-based

Operates on token frequency vectors (Counter), more sensitive to repeated terms and extra content:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Cosine** | `(A·B) / (\|A\|·\|B\|)` | Frequency-distribution similarity; high even when extra content is present |
| **Jaccard** | `\|A ∩ B\| / \|A ∪ B\|` | Set overlap over union; penalises both extra and missing tokens |
| **Excess Ratio** | `1 − Σ min(fp[t], fg[t]) / Σ fp[t]` | Fraction of extracted tokens not covered by gold — **lower is better** |

---

## Evaluation Results

Average scores across all gold standard URLs per domain (`/full_gs_eval`).

### Token Level Eval

| Domain | Precision | Recall | F1 |
|--------|-----------|--------|----|
| it.wikipedia.org | 0.9838 | 0.9778 | 0.9808 |
| www.cnbc.com | 0.9988 | 0.9999 | 0.9993 |
| www.espn.com | 0.9993 | 0.9981 | 0.9987 |
| www.xe.com | 0.9916 | 0.9935 | 0.9925 |

### Similarity Eval

| Domain | Cosine | Jaccard | Excess Ratio |
|--------|--------|---------|--------------|
| it.wikipedia.org | 0.9949 | 0.9628 | 0.0116 |
| www.cnbc.com | 1.0000 | 0.9987 | 0.0006 |
| www.espn.com | 0.9802 | 0.9974 | 0.0185 |
| www.xe.com | 0.9984 | 0.9852 | 0.0065 |

---

## Grader

Computer Engineering Laboratory — A.Y. 2025/2026

The grader (`lab-grader.tar.gz`, image tag `lab-grader-progetto-finale:1.0.4`) tests the four project components (backend, database, Ollama, frontend) through every REST endpoint. It expects the stack on its default ports: backend `8003`, frontend `8004`, MariaDB `3306`, Ollama `11434`.

The database is mutated during the run; reset the stack between executions if you re-run the grader. The project is graded exactly as delivered: verify the grader passes immediately after extracting the archive, with no further changes.

```bash
make up
make grader-load
make test STUDENT_ID=<your_student_id>
```

Equivalent raw commands:

```bash
docker load -i lab-grader.tar.gz
docker run --network host lab-grader-progetto-finale:1.0.4 <your_student_id>

# With JSON report:
docker run --network host \
  -v "$(pwd)/output:/output" \
  lab-grader-progetto-finale:1.0.4 <your_student_id> --machine -o /output/report.json
```
