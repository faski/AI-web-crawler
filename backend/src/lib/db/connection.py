"""MariaDB connection pool with startup retry.

Configuration is read from environment variables:
    DB_HOST     hostname  (default: localhost)
    DB_PORT     port      (default: 3306)
    DB_NAME     database  (default: creeping_crawler)
    DB_USER     username  (default: creeping_crawler)
    DB_PASSWORD password  (default: creeping_crawler)

The pool is created lazily on the first ``init_pool()`` call.
"""

import os
import time

import mariadb

CONNECTION_POOL_SIZE = 5
CONNECTION_POOL_NAME = "creeping-crawler-pool"

_connection_pool: mariadb.ConnectionPool | None = None


def _database_config() -> dict:
    """Read DB_* settings from the environment."""
    return {
        "host": os.environ.get("DB_HOST", "localhost"),
        "port": int(os.environ.get("DB_PORT", "3306")),
        "user": os.environ.get("DB_USER", "creeping_crawler"),
        "password": os.environ.get("DB_PASSWORD", "creeping_crawler"),
        "database": os.environ.get("DB_NAME", "creeping_crawler"),
    }


def init_pool(max_attempts: int = 60, delay_seconds: float = 2.0) -> None:
    """Create the connection pool, retrying until MariaDB is reachable.

    Called once at FastAPI startup. The backend container can start before
    MariaDB is ready, so we poll the server until it accepts connections.
    """
    global _connection_pool
    if _connection_pool is not None:
        return

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            _connection_pool = mariadb.ConnectionPool(
                pool_name=CONNECTION_POOL_NAME,
                pool_size=CONNECTION_POOL_SIZE,
                pool_reset_connection=False,
                **_database_config(),
            )
            return
        except mariadb.Error as error:
            last_error = error
            print(
                f"[db] connection attempt {attempt}/{max_attempts} failed: {error}",
                flush=True,
            )
            time.sleep(delay_seconds)

    raise RuntimeError(f"MariaDB unreachable after {max_attempts} attempts: {last_error}")


def close_pool() -> None:
    """Close the pool on application shutdown."""
    global _connection_pool
    if _connection_pool is not None:
        _connection_pool.close()
        _connection_pool = None


def get_connection() -> mariadb.Connection:
    """Take a connection from the pool."""
    if _connection_pool is None:
        raise RuntimeError("Connection pool not initialized; call init_pool() first")
    return _connection_pool.get_connection()


def fetch_one(sql: str, params: tuple = ()):
    """Run a SELECT and return the first row (or None)."""
    connection = get_connection()
    query = connection.cursor()
    query.execute(sql, params)
    row = query.fetchone()
    connection.close()
    return row


def fetch_all(sql: str, params: tuple = ()) -> list:
    """Run a SELECT and return all the rows."""
    connection = get_connection()
    query = connection.cursor()
    query.execute(sql, params)
    rows = query.fetchall()
    connection.close()
    return rows


def execute(sql: str, params: tuple = ()) -> int:
    """Run an INSERT/UPDATE/DELETE, save the changes and return how many rows changed."""
    connection = get_connection()
    command = connection.cursor()
    command.execute(sql, params)
    connection.commit()
    changed = command.rowcount
    connection.close()
    return changed


def ping() -> bool:
    """Return True if the database is reachable, False otherwise.

    Used by the /status endpoint to report database health without raising.
    """
    if _connection_pool is None:
        return False
    try:
        connection = get_connection()
        connection.ping()
        connection.close()
        return True
    except mariadb.Error:
        return False
