import json
import httpx
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import REGISTRY
import sureshot.alerts as alerts
import sureshot.cli as cli
from sureshot import Store, calibrate, review_queue
from sureshot.adapters.base import Outage
from sureshot.router import Router
from test_router import QUESTIONS, FakeClaude, Tier

SPANS = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(SPANS))
trace.set_tracer_provider(_provider)


def value(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


class Down:
    def decide(self, state, questions, images=()):
        raise Outage("overloaded")


class Wrong:
    def decide(self, state, questions, images=()):
        raise ValueError("bad key")


def test_metrics_count_answers_latency_cost_and_outages(monkeypatch):
    monkeypatch.setenv("SURESHOT_COSTS", '{"tier 0": 0.001, "tier 1": 0.01, "system2": 0.1}')
    q = {"m_sure": {"type": "noul"}, "m_unsure": {"type": "noul"}}
    before = {"sure": value("sureshot_answers_total", question="m_sure", by="tier 0"), "s2": value("sureshot_answers_total", question="m_unsure", by="system2"),
              "decisions": value("sureshot_decisions_total"), "total": value("sureshot_stage_seconds_count", stage="total"),
              "s2_seconds": value("sureshot_stage_seconds_count", stage="system2"),
              "cost0": value("sureshot_cost_total", question="m_unsure", stage="tier 0"), "cost2": value("sureshot_cost_total", question="m_unsure", stage="system2"),
              "outage": value("sureshot_errors_total", stage="tier 0", kind="outage"), "error": value("sureshot_errors_total", stage="decide", kind="error")}
    router = Router([Tier({"m_sure": (True, 0.99), "m_unsure": (True, 0.5)})], FakeClaude())
    for _ in range(3):
        router.decide("x", q)
    assert value("sureshot_decisions_total") - before["decisions"] == 3
    assert value("sureshot_answers_total", question="m_sure", by="tier 0") - before["sure"] == 3
    assert value("sureshot_answers_total", question="m_unsure", by="system2") - before["s2"] == 3
    assert value("sureshot_stage_seconds_count", stage="total") - before["total"] == 3
    assert value("sureshot_stage_seconds_count", stage="system2") - before["s2_seconds"] == 3
    assert value("sureshot_cost_total", question="m_unsure", stage="tier 0") - before["cost0"] == pytest.approx(0.003)
    assert value("sureshot_cost_total", question="m_unsure", stage="system2") - before["cost2"] == pytest.approx(0.3)

    Router([Down(), Tier({"m_sure": (True, 0.99), "m_unsure": (True, 0.99)})], FakeClaude(), retries=0).decide("x", q)
    assert value("sureshot_errors_total", stage="tier 0", kind="outage") - before["outage"] == 1
    with pytest.raises(ValueError):
        Router(Wrong(), FakeClaude(), retries=0).decide("x", q)
    assert value("sureshot_errors_total", stage="decide", kind="error") - before["error"] == 1


def test_metrics_endpoint(monkeypatch):
    from fastapi.testclient import TestClient
    import sureshot.api as api

    monkeypatch.setenv("SURESHOT_API_KEY", "secret")
    monkeypatch.setattr(api, "_router", Router(Tier({"m_api": (True, 0.99)}), FakeClaude()))
    client = TestClient(api.app)
    client.post("/decide", json={"state": "x", "questions": {"m_api": {"type": "noul"}}}, headers={"Authorization": "Bearer secret"})
    assert client.get("/metrics").status_code == 401
    response = client.get("/metrics", headers={"Authorization": "Bearer secret"})
    assert response.headers["content-type"].startswith("text/plain")
    assert 'sureshot_answers_total{by="tier 0",question="m_api"}' in response.text


def test_spans_around_each_tier_and_system2():
    SPANS.clear()
    q = {"t_sure": {"type": "noul"}, "t_unsure": {"type": "noul"}}
    Router([Down(), Tier({"t_sure": (True, 0.99), "t_unsure": (True, 0.5)})], FakeClaude(), retries=0).decide("x", q)
    spans = {s.name + str(s.attributes.get("sureshot.tier", "")): s for s in SPANS.get_finished_spans()}
    assert set(spans) == {"sureshot.decide", "sureshot.tier0", "sureshot.tier1", "sureshot.system2"}
    root = spans["sureshot.decide"]
    assert all(s.parent.span_id == root.context.span_id for name, s in spans.items() if name != "sureshot.decide")
    assert spans["sureshot.tier0"].status.status_code == trace.StatusCode.ERROR and spans["sureshot.tier0"].events[0].name == "exception"
    assert spans["sureshot.tier1"].attributes["sureshot.questions"] == 2
    assert spans["sureshot.system2"].attributes["sureshot.questions"] == 1
    assert root.attributes["sureshot.escalated"] == 1


@pytest.fixture
def sent(monkeypatch):
    posts = []

    def post(url, json, timeout):
        posts.append((url, json))
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(alerts.httpx, "post", post)
    return posts


def _certified(tmp_path):
    store = Store(tmp_path / "s.db", seed=0)
    q = {"needs_prep": QUESTIONS["needs_prep"]}
    router = Router(Tier({"needs_prep": (True, 0.99)}), FakeClaude(), log=store, audit=1.0)
    for _ in range(300):
        router.decide("event", q)
    assert calibrate(store, 0.9, 30)["needs_prep"]["threshold"] == 0.99
    return store


def test_a_failed_recheck_alerts_once(tmp_path, monkeypatch, capsys, sent):
    store = _certified(tmp_path)
    db = str(tmp_path / "s.db")
    cli.main(["recheck", "--db", db, "--sample", "0.1"])
    assert sent == []  # no URL, nothing sent
    monkeypatch.setenv("SURESHOT_ALERT_URL", "https://alerts.example/hook")
    cli.main(["recheck", "--db", db, "--sample", "0.1"])
    assert sent == []  # nothing failed
    for item in review_queue(store, limit=None, sample=0.1):
        store.label(item["id"], "needs_prep", False)
    cli.main(["recheck", "--db", db, "--sample", "0.1"])
    assert sent == [("https://alerts.example/hook", {"event": "recheck", "items": [{"tier": 0, "key": "needs_prep", "threshold": 0.99}],
                                                      "text": "sureshot: on tier 0, needs_prep failed recheck against your labels; System 2 answers them until a calibration certifies a threshold again"})]
    cli.main(["recheck", "--db", db, "--sample", "0.1"])
    assert len(sent) == 1


def test_a_demotion_alerts_slack_with_text_only(tmp_path, monkeypatch, capsys, sent):
    store = _certified(tmp_path)
    monkeypatch.setenv("SURESHOT_ALERT_URL", "https://hooks.slack.com/services/T/B/x")
    for item in review_queue(store, limit=None, sample=0.1):
        store.label(item["id"], "needs_prep", False)
    cli.main(["calibrate", "--db", str(tmp_path / "s.db"), "--target", "0.9", "--sample", "0.1"])
    assert store.thresholds() == {"needs_prep": float("inf")}
    assert sent == [("https://hooks.slack.com/services/T/B/x", {"text": "sureshot: on tier 0, needs_prep were demoted; System 2 answers them until a calibration certifies a threshold again"})]


def test_the_api_calibrate_alerts_on_a_demotion(tmp_path, monkeypatch, sent):
    from fastapi.testclient import TestClient
    import sureshot.api as api

    store = _certified(tmp_path)
    for item in review_queue(store, limit=None, sample=0.1):
        store.label(item["id"], "needs_prep", False)
    monkeypatch.setenv("SURESHOT_API_KEY", "secret")
    monkeypatch.setenv("SURESHOT_ALERT_URL", "https://alerts.example/hook")
    monkeypatch.setattr(api, "_router", Router(Tier({}), FakeClaude(), log=store))
    response = TestClient(api.app).post("/calibrate", json={"target_accuracy": 0.9, "sample": 0.1}, headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    assert [body["event"] for _, body in sent] == ["demotion"]


def test_drift_alerts_and_a_failed_delivery_does_not_fail_the_job(monkeypatch, capsys):
    report = {"drift": [{"tier": 0, "key": "a", "flags": ["escalation_rate"], "change": {}}, {"tier": 0, "key": "b", "flags": [], "change": {}}]}
    monkeypatch.setenv("SURESHOT_ALERT_URL", "https://alerts.example/hook")

    def down(url, json, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(alerts.httpx, "post", down)
    alerts.drift(report)
    assert "could not send the drift alert" in capsys.readouterr().err
    posts = []
    monkeypatch.setattr(alerts.httpx, "post", lambda url, json, timeout: posts.append(json) or httpx.Response(200, request=httpx.Request("POST", url)))
    alerts.drift(report)
    assert posts == [{"event": "drift", "text": "sureshot: drift on tier 0 a (escalation_rate)", "items": [{"tier": 0, "key": "a", "flags": ["escalation_rate"], "change": {}}]}]
    alerts.drift({"drift": report["drift"][1:]})
    assert len(posts) == 1
