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
