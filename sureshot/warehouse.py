"""SQL functions that call sureshot: a BigQuery remote function or a Snowflake external function sends a batch of
rows, each (question set, text), and gets one JSON object per row, with the same fields as `decide --batch` writes
(see batch.columns). A row that can't be decided (no text, an unknown set) gets {"sureshot_error": ...} rather than
failing the query; anything else, such as System 1 being down, fails the batch so the warehouse retries it.

Warehouses retry batches, so a row can be decided and logged more than once; with the answer cache on, the models
are only asked once."""
from concurrent.futures import ThreadPoolExecutor
from sureshot.batch import columns

WORKERS = 8


def decide(router, calls):
    """One result per (question set, text) call, decided in parallel, with System 2 asked directly."""
    def one(call):
        name, text = call
        if not isinstance(text, str) or not text.strip():
            return {"sureshot_error": "no text"}
        try:
            questions = router.question_set(name)
            return columns(questions, router.decide(text, questions))
        except ValueError as e:
            return {"sureshot_error": str(e)}

    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(one, calls))
