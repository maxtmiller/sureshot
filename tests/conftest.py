"""With SURESHOT_TEST_POSTGRES set to a Postgres URL, every test runs against Postgres: each SQLite path a test opens
becomes its own schema in that database, dropped afterwards. Tests marked sqlite check SQLite itself and are skipped."""
import hashlib
import os
from pathlib import Path
import pytest

URL = os.environ.get("SURESHOT_TEST_POSTGRES")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if URL and "sqlite" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="checks SQLite itself"))


@pytest.fixture(autouse=True)
def postgres_instead_of_sqlite(monkeypatch):
    if not URL:
        yield
        return
    import psycopg
    import sureshot.db as db

    schemas, opened = set(), set(db._pools)

    def database(location):
        if db.is_postgres(location):
            return db.Postgres(location)
        Path(location).touch()  # so a check that the log exists, as in the CLI, still sees it
        schema = "t_" + hashlib.sha1(str(location).encode()).hexdigest()[:16]
        if schema not in schemas:
            with psycopg.connect(URL, autocommit=True) as conn:
                conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
                conn.execute(f"CREATE SCHEMA {schema}")
            schemas.add(schema)
        return db.Postgres(f"{URL}{'&' if '?' in URL else '?'}options=-csearch_path%3D{schema}")

    monkeypatch.setattr(db, "database", database)
    monkeypatch.setattr("sureshot.store.database", database)
    monkeypatch.setattr("sureshot.cache.database", database)
    yield
    for url in set(db._pools) - opened:
        db._pools.pop(url).close()
    with psycopg.connect(URL, autocommit=True) as conn:
        for schema in schemas:
            conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
