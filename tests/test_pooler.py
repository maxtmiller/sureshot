"""Through a connection pooler in transaction mode (SURESHOT_TEST_POOLER: a postgresql:// URL to, say, PgBouncer with
max_prepared_statements = 0, as hosted Postgres's pooled URLs often are), each transaction may run on a different
server connection."""
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
import pytest
from sureshot import Cache, Router, Store, calibrate

URL = os.environ.get("SURESHOT_TEST_POOLER")
pytestmark = pytest.mark.skipif(not URL, reason="set SURESHOT_TEST_POOLER to a pooled Postgres URL")


def test_the_log_and_cache_work_through_a_pooler():
    key = f"ok_{uuid.uuid4().hex[:8]}"
    question = {key: {"type": "noul"}}

    class Sure:
        identity = key

        def decide(self, state, questions, images=()):
            return {k: {"answer": True, "confidence": 0.99, "probabilities": {}} for k in questions}

    store = Store(URL)
    router = Router(Sure(), log=store, cache=Cache(URL))
    before = len(store.results())

    def work(i):
        store.label(router.decide(f"{key} {i}", question)["id"], key, True)

    # Each query runs far more often than psycopg's threshold for preparing it, across many server connections.
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(work, range(200)))
    assert len(store.results()) - before == 200
    assert calibrate(store, 0.9, 30)[key]["threshold"] == 0.99
    assert router.decide(f"{key} 0", question)["cache"]["hits"] == 1
