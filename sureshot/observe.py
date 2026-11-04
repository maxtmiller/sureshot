"""Prometheus metrics and OpenTelemetry spans for every decision the Router makes, in whatever process makes it.

With prometheus-client installed (it comes with sureshot[api]), each process counts its decisions, answers (by
question and by who answered: "tier N" or "system2"), latency by stage, outages and errors, and with SURESHOT_COSTS
(the JSON `sureshot stats --costs` takes) what each stage cost. The API serves them at GET /metrics, and
`sureshot worker --metrics-port` serves them the same way with serve().

With opentelemetry-api installed (sureshot[otel]) and an OpenTelemetry SDK configured, each decide is a span
"sureshot.decide" with a child per tier, the guard and System 2. Without it, both cost nothing."""
import hmac
import json
import os
import threading
from contextlib import contextmanager
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
except ImportError:
    Counter = None

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    trace = None

if Counter is not None:
    DECISIONS = Counter("sureshot_decisions", "Decisions made")
    ANSWERS = Counter("sureshot_answers", "Answers, by question and by who answered (tier N or system2)", ["question", "by"])
    LATENCY = Histogram("sureshot_stage_seconds", "Time spent in each stage of a decision (tier N, system2, total)", ["stage"],
                        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60))
    COST = Counter("sureshot_cost", "What each stage cost, at the SURESHOT_COSTS prices per question asked", ["question", "stage"])
    ERRORS = Counter("sureshot_errors", "Outages a decision fell back from (outage) and errors it raised (error), by stage", ["stage", "kind"])


def available():
    return Counter is not None


def latest():
    """(body, content type) of the metrics in the Prometheus text format."""
    return generate_latest(), CONTENT_TYPE_LATEST


def serve(port, key, host="0.0.0.0"):
    """Serves GET /metrics with the bearer token key on a background thread, for a process without the API."""
    class Metrics(BaseHTTPRequestHandler):
        def do_GET(self):
            if urlparse(self.path).path != "/metrics":
                return self.reply(404, b"not found\n")
            if not hmac.compare_digest(self.headers.get("Authorization", "").encode(), f"Bearer {key}".encode()):
                return self.reply(401, b"missing or wrong bearer token\n")
            self.reply(200, *latest())

        def reply(self, status, body, content_type="text/plain; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Metrics)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@lru_cache(maxsize=4)
def _prices(text):
    costs = json.loads(text)
    if not isinstance(costs, dict) or not all(isinstance(v, (int, float)) for v in costs.values()):
        raise ValueError('SURESHOT_COSTS must map each stage to a cost per question, e.g. {"tier 0": 0, "system2": 0.004}')
    return costs


def _stage(degraded):
    return "system2" if degraded["system"] == 2 else "guard" if degraded["system"] == "guard" else f"tier {degraded['tier']}"


def decided(result, asked):
    """Counts a decision (the Router's result); asked is {stage: [keys]} of the questions each stage was asked."""
    if Counter is None:
        return
    DECISIONS.inc()
    for key, answer in result["answers"].items():
        ANSWERS.labels(key, "system2" if answer["source"] == "system2" else f"tier {answer['tier']}").inc()
    latency = result["latency_ms"]
    for i, ms in enumerate(latency["tiers"]):
        LATENCY.labels(f"tier {i}").observe(ms / 1000)
    if latency["system2"] is not None:
        LATENCY.labels("system2").observe(latency["system2"] / 1000)
    LATENCY.labels("total").observe(latency["total"] / 1000)
    for degraded in result["degraded"] or []:
        ERRORS.labels(_stage(degraded), "outage").inc()
    if os.environ.get("SURESHOT_COSTS"):
        prices = _prices(os.environ["SURESHOT_COSTS"])
        for stage, keys in asked.items():
            for key in keys:
                COST.labels(key, stage).inc(prices.get(stage, 0.0))


def failed():
    """Counts a decision that raised."""
    if Counter is not None:
        ERRORS.labels("decide", "error").inc()


class _NoSpan:
    def set_attribute(self, *args):
        pass

    def outage(self, error):
        pass


class _Span:
    def __init__(self, span):
        self.span = span

    def set_attribute(self, key, value):
        self.span.set_attribute(key, value)

    def outage(self, error):
        """A model that was down, which the decision fell back from instead of raising."""
        self.span.record_exception(error)
        self.span.set_status(Status(StatusCode.ERROR, repr(error)))


@contextmanager
def span(name, **attributes):
    """A span named name with sureshot.<attribute> attributes, or a stand-in that does nothing without OpenTelemetry."""
    if trace is None:
        yield _NoSpan()
        return
    with trace.get_tracer("sureshot").start_as_current_span(name, attributes={f"sureshot.{k}": v for k, v in attributes.items()}) as s:
        yield _Span(s)
