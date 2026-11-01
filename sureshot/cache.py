"""A response cache: what a model answered about a text, so asking again costs nothing."""
import hashlib
import json
import time
from sureshot.db import create, database


def cache_key(identity, key, question, state, images):
    """The same model (identity), question name and definition, state and images always give the same key."""
    return hashlib.sha256(json.dumps([identity, key, question, state, list(images)], sort_keys=True).encode()).hexdigest()


class Cache:
    """Key-value store of model answers, in a SQLite file or a Postgres database (a postgresql:// URL, which replicas can
    share). ttl is how many seconds an answer stays valid (None: forever).

    Router(cache=Cache(path)) caches per question and per model: a model's identity (its name, URL, extra request
    fields, prompt version, or a fingerprint of a trained classifier) is part of the key, so changing the model never
    serves the old model's answers. A model with no `identity` is not cached."""

    def __init__(self, path, ttl=None):
        self.path, self.ttl, self._database = str(path), ttl, database(path)
        with self._db() as db:
            create(db, self._database.postgres, "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, ts DOUBLE PRECISION, value TEXT)")
            if not self._database.postgres:
                db.execute("PRAGMA journal_mode=WAL")

    def _db(self):
        return self._database.connect()

    def get_many(self, keys):
        """{key: value} for the keys that are cached and not expired."""
        keys = list(keys)
        if not keys:
            return {}
        oldest = time.time() - self.ttl if self.ttl is not None else 0.0
        with self._db() as db:
            rows = db.execute(f"SELECT key, value FROM cache WHERE ts >= ? AND key IN ({','.join('?' * len(keys))})", [oldest, *keys]).fetchall()
        return {key: json.loads(value) for key, value in rows}

    def put_many(self, items):
        now = time.time()
        with self._db() as db:
            db.executemany("INSERT INTO cache VALUES (?, ?, ?) ON CONFLICT (key) DO UPDATE SET ts = excluded.ts, value = excluded.value",
                           [(key, now, json.dumps(value)) for key, value in items.items()])

    def size(self):
        with self._db() as db:
            return db.execute("SELECT COUNT(*) FROM cache").fetchone()[0]

    def clear(self):
        with self._db() as db:
            db.execute("DELETE FROM cache")
