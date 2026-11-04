"""Copy a SQLite log (or cache) to Postgres: `sureshot migrate --from sureshot.db --to postgresql://...` copies the
decisions, labels, thresholds and calibrations, and the cache when the file has one. Running it again copies only the
rows the target lacks: a row already there is kept, never overwritten."""
import re
import sqlite3
from pathlib import Path
from sureshot.cache import Cache
from sureshot.store import SCHEMA, Store

TABLES = re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", SCHEMA) + ["cache"]
CHUNK = 1000


def migrate(source, target):
    """Returns {table: {rows: in the source, copied: new in the target}} for each sureshot table the source has."""
    if not Path(source).is_file():
        raise ValueError(f"no SQLite file at {source}")
    src = sqlite3.connect(f"file:{Path(source).resolve()}?mode=ro", uri=True)
    try:
        present = {name for (name,) in src.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        tables = [t for t in TABLES if t in present]
        if not tables:
            raise ValueError(f"{source} has no sureshot tables")
        store = Store(target)
        if "cache" in tables:
            Cache(target)
        report = {}
        for table in tables:
            # Only the columns the source has: a log made by an older sureshot lacks some, which stay empty.
            columns = [row[1] for row in src.execute(f"PRAGMA table_info({table})")]
            insert = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))}) ON CONFLICT DO NOTHING"
            count = f"SELECT COUNT(*) FROM {table}"
            with store._db() as db:
                before = db.execute(count).fetchone()[0]
            rows = src.execute(f"SELECT {', '.join(columns)} FROM {table}")
            while chunk := rows.fetchmany(CHUNK):
                with store._db() as db:
                    db.executemany(insert, chunk)
            with store._db() as db:
                after = db.execute(count).fetchone()[0]
            report[table] = {"rows": src.execute(count).fetchone()[0], "copied": after - before}
        return report
    finally:
        src.close()
