"""Where Store and Cache keep their tables: a SQLite file, or a Postgres database for several processes or machines
that share one log. Both take the same SQL, with ? placeholders and ON CONFLICT upserts."""
import atexit
import sqlite3
import threading
from contextlib import contextmanager

_pools, _pools_lock = {}, threading.Lock()


@atexit.register
def _close_pools():
    """Left open, each pool's threads hold up the interpreter's exit by 5 seconds apiece."""
    for pool in _pools.values():
        pool.close()


def is_postgres(location):
    return str(location).startswith(("postgresql://", "postgres://"))


class SQLite:
    postgres = False

    def __init__(self, path):
        self.path = str(path)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()


class _Connection:
    """The sqlite3 calls the store makes (execute and executemany, with ? placeholders) on a psycopg connection."""

    def __init__(self, conn):
        self.conn = conn

    def execute(self, sql, params=()):
        return self.conn.execute(sql.replace("?", "%s"), params)

    def executemany(self, sql, rows):
        with self.conn.cursor() as cursor:
            cursor.executemany(sql.replace("?", "%s"), rows)


class Postgres:
    """A pool of connections to a Postgres URL, shared by every Store and Cache in the process that uses that URL."""
    postgres = True

    def __init__(self, url):
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as e:
            raise ImportError('A Postgres log needs psycopg: pip install "sureshot[postgres]"') from e
        with _pools_lock:
            if url not in _pools:
                # No prepared statements: through a pooler in transaction mode (PgBouncer, Supabase's and Neon's pooled
                # URLs) the next transaction may get another server connection, where they collide or are missing.
                _pools[url] = ConnectionPool(url, min_size=1, max_size=10, open=True, kwargs={"prepare_threshold": None})
            self.pool = _pools[url]

    @contextmanager
    def connect(self):
        with self.pool.connection() as conn:  # commits when the block succeeds, rolls back when it raises
            yield _Connection(conn)


def database(location):
    return Postgres(location) if is_postgres(location) else SQLite(location)


def create(db, postgres, schema):
    """Runs each statement of a schema. On Postgres a lock makes processes that start together take turns, since two
    CREATE TABLE IF NOT EXISTS of the same table at once can still collide there."""
    if postgres:
        db.execute("SELECT pg_advisory_xact_lock(7297)")
    for statement in schema.split(";"):
        if statement.strip():
            db.execute(statement)
