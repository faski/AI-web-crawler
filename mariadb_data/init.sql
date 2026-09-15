-- Schema iniziale di Creeping Crawler.
-- Eseguito automaticamente da MariaDB al primo avvio del container.

USE creeping_crawler;

-- Tabella web_resources
CREATE TABLE IF NOT EXISTS web_resources (
    url VARCHAR(2048) CHARACTER SET ascii NOT NULL,
    domain VARCHAR(255) NOT NULL,
    title VARCHAR(2048),
    html_text LONGTEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (url)
);

-- Tabella gold_standard
CREATE TABLE IF NOT EXISTS gold_standard (
    url VARCHAR(2048) CHARACTER SET ascii NOT NULL,
    gold_text LONGTEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (url),
    FOREIGN KEY (url) REFERENCES web_resources(url) ON DELETE CASCADE
);

-- Tabella evaluations: cache delle metriche per /db_stats e /full_gs_eval,
-- popolata progressivamente dagli endpoint di evaluation.
CREATE TABLE IF NOT EXISTS evaluations (
    url VARCHAR(2048) CHARACTER SET ascii NOT NULL,
    precision_val DOUBLE,
    recall_val DOUBLE,
    f1 DOUBLE,
    cosine DOUBLE,
    jaccard DOUBLE,
    excess_ratio DOUBLE,
    judge_score INT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (url),
    FOREIGN KEY (url) REFERENCES web_resources(url) ON DELETE CASCADE
);

-- Tabelle delle run del parser LLM (modifica 3a).
-- Una run = un passaggio del parser LLM sul gold standard con un dato modello
-- e una data condizione. Le pagine di confronto leggono solo da qui, cosi'
-- guardare i risultati non costa mai una chiamata al modello.
CREATE TABLE IF NOT EXISTS llm_runs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    label VARCHAR(191) CHARACTER SET ascii NOT NULL,
    model_name VARCHAR(255) CHARACTER SET ascii NOT NULL,
    provider VARCHAR(32) CHARACTER SET ascii NOT NULL,
    condition_name VARCHAR(64) NOT NULL,
    budget_tokens INT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_llm_run_label (label)
);

-- url in utf8mb4 ma corto: la chiave (run_id, url) pesa 2804 byte, sotto il
-- limite di 3072 che InnoDB impone a un indice. Non ascii, perche' il gold
-- standard contiene URL accentati e una colonna ascii li sostituirebbe con '?'.
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
);
