"""Decide every row of a file at once (`sureshot decide --batch`), for backfills and nightly jobs.

System 1 answers each row, several at a time. When System 2 has a batch API (OpenAI's, or Anthropic's Message
Batches, at half the price), the questions it must answer are sent as batches and waited for; otherwise it is asked
directly, as a live decision would. Each row is then finished and logged like any decision, so a batch run feeds the
review queue and calibration as live traffic does.

Progress is kept in a file next to the output, one line per step, so running the same command again after it stops
picks up where it was: rows already decided are not asked again, and batches already sent are waited for, not sent
again. The file is removed once the output is written."""
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from sureshot.cache import cache_key
from sureshot.router import _transient

PER_BATCH = 10_000  # well under OpenAI's 50,000 requests and Anthropic's 100,000 per batch


def decide_rows(router, questions, data, progress, text="text", escalate=True, workers=8, poll=60, say=lambda message: None):
    """The result of each row of `data` (dicts with a `text` field), in order: a decision, or {"error": ...} for a
    row with no text or one no model could answer. progress is the path of the progress file; say(message) is told
    how it goes."""
    questions = router.question_set(questions)
    progress = Path(progress)
    header = {"rows": len(data), "text": text, "questions": hashlib.sha256(json.dumps(questions, sort_keys=True).encode()).hexdigest()}
    done, pending, sent = {}, {}, []
    if progress.exists():
        lines = progress.read_text().splitlines()
        if not lines or json.loads(lines[0]) != {"header": header}:
            raise ValueError(f"{progress} is from a run on other rows or questions; delete it to start over")
        for line in lines[1:]:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                break  # the last line, cut off when the run stopped
            if "pending" in entry:
                pending[entry["n"]] = entry["pending"]
            elif "batch" in entry:
                sent.append(entry)
            else:
                done[entry["n"]] = entry["result"]
                pending.pop(entry["n"], None)
        say(f"resuming: {len(done)} rows decided, {len(pending)} waiting for System 2")
    else:
        progress.write_text(json.dumps({"header": header}) + "\n")
    lock = threading.Lock()
    journal = progress.open("a")

    def note(entry):
        with lock:
            journal.write(json.dumps(entry) + "\n")
            journal.flush()

    def finished(n, result):
        done[n] = result
        note({"n": n, "result": result})

    system2 = router.system2 if escalate else None
    batching = bool(getattr(system2, "batches", False))

    def first(n):
        content = data[n].get(text)
        if not isinstance(content, str) or not content.strip():
            return n, {"error": f"no {text!r}"}, None
        try:
            if not batching:
                return n, router.decide(content, questions, escalate=escalate), None
            start = time.perf_counter()
            p = router.first(content, questions, escalate=escalate)
            p["ms"] = (time.perf_counter() - start) * 1000
            return (n, None, p) if p["ask"] else (n, router.finish(p, None, total_ms=p["ms"]), None)
        except ValueError as e:
            return n, {"error": str(e)}, None

    todo = [n for n in range(len(data)) if n not in done and n not in pending]
    with ThreadPoolExecutor(max(1, workers)) as pool:
        for count, (n, result, p) in enumerate(pool.map(first, todo), 1):
            if p is None:
                finished(n, result)
            else:
                pending[n] = p
                note({"n": n, "pending": p})
            if count % 100 == 0:
                say(f"System 1: {count}/{len(todo)} rows")

    if pending:
        got = _batches(router, system2, pending, sent, note, poll, say) if batching else {}
        def last(n):
            try:
                return n, _finish(router, system2, pending[n], got.get(n, {}))
            except ValueError as e:
                return n, {"error": str(e)}
        with ThreadPoolExecutor(max(1, workers)) as pool:
            for n, result in pool.map(last, sorted(pending)):
                finished(n, result)
    journal.close()
    return [done[n] for n in range(len(data))]


def _batches(router, system2, pending, sent, note, poll, say):
    """{row: System 2's answers} from batches: the questions not yet sent (nor cached) go out first, then every
    batch is waited for."""
    covered = {custom_id for entry in sent for custom_id in entry["asked"]}
    items = []
    for n, p in sorted(pending.items()):
        custom_id = f"row-{n}"
        keys = [key for key in p["ask"] if not _cached(router, system2, p["state"], key, p["ask"][key])]
        if keys and custom_id not in covered:
            items.append((custom_id, p["state"], {key: p["ask"][key] for key in keys}))
    for i in range(0, len(items), PER_BATCH):
        chunk = items[i:i + PER_BATCH]
        entry = {"batch": router._call(system2.batch_submit, chunk), "asked": {custom_id: list(asked) for custom_id, _, asked in chunk}}
        note(entry)
        sent.append(entry)
    say(f"System 2: {sum(len(entry['asked']) for entry in sent)} rows in {len(sent)} batches")
    got, waiting = {}, list(sent)
    while waiting:
        for entry in list(waiting):
            asked = {custom_id: {key: pending[int(custom_id[4:])]["ask"][key] for key in keys} for custom_id, keys in entry["asked"].items()}
            try:
                results = router._call(system2.batch_results, entry["batch"], asked)
            except Exception as e:
                if not _transient(e):
                    raise
                say(f"System 2: could not check a batch ({e!r}); trying again")  # a batch waits up to 24 hours: a blip must not end the run
                results = None
            if results is not None:
                waiting.remove(entry)
                got |= {int(custom_id[4:]): answers for custom_id, answers in results.items()}
        if waiting:
            say(f"System 2: waiting for {len(waiting)} of {len(sent)} batches")
            time.sleep(poll)
    if router.cache and getattr(system2, "identity", None):
        router.cache.put_many({cache_key(system2.identity, key, pending[n]["ask"][key], pending[n]["state"], ()): answer
                               for n, answers in got.items() for key, answer in answers.items()})
    return got


def _cached(router, system2, state, key, question):
    identity = getattr(system2, "identity", None) if router.cache else None
    return identity is not None and bool(router.cache.get_many([cache_key(identity, key, question, state, ())]))


def _finish(router, system2, p, answers):
    """A pending row's decision, with System 2's batch answers; what a batch did not answer (it was cached, failed
    or declined) is asked directly, as a live decision would."""
    rest = {key: question for key, question in p["ask"].items() if key not in answers}
    slow, outage = dict(answers), None
    if rest:
        try:
            slow |= router._cached(system2, "answer", p["state"], rest, (), p["cache"])
        except Exception as e:
            if not _transient(e) or any(key not in p["latest"] for key in p["escalated"]):
                raise
            slow, outage = None, repr(e)
    return router.finish(p, slow, outage, total_ms=p["ms"])


def columns(questions, result):
    """The output columns for one row's result: sureshot_id, then each question's answer, source and confidence."""
    if "error" in result:
        return {"sureshot_error": result["error"]}
    out = {"sureshot_id": result.get("id")}
    for key, answer in result["answers"].items():
        out[f"sureshot_{key}"] = answer["answer"]
        out[f"sureshot_{key}_source"] = "system2" if answer["source"] == "system2" else f"tier {answer['tier']}"
        out[f"sureshot_{key}_confidence"] = answer.get("confidence")
    return out
