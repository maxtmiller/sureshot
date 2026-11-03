import json
import pytest
from sureshot import Store, worker
from sureshot.router import Router
from test_router import QUESTIONS, FakeClaude, FakeDecider


class Queue:
    """An in-memory source and sink."""

    def __init__(self, *messages):
        self.waiting, self.sent, self.acked = [json.dumps(m).encode() if not isinstance(m, bytes) else m for m in messages], [], []

    def receive(self, n):
        got, self.waiting = self.waiting[:n], self.waiting[n:]
        return [(m, m) for m in got]

    def ack(self, handles):
        self.acked += handles

    def send(self, messages):
        self.sent += [json.loads(m) for m in messages]


def _run(router, source, sink=None, dead=None, **kw):
    sink, dead = sink or Queue(), dead or Queue()
    batches = []
    result = worker.run(router, source, sink, dead, "events", workers=2, backoff=0, stop=lambda: batches.append(1) or len(batches) > 3, **kw)
    return result, sink, dead


@pytest.fixture
def router(tmp_path):
    return Router(FakeDecider(), FakeClaude(), log=Store(tmp_path / "sureshot.db"), questions={"events": QUESTIONS})


def test_messages_are_decided_sent_and_then_acknowledged(router):
    source = Queue({"state": "standup", "ref": "cal-1"}, {"state": "interview", "questions": "events"}, b"not json", {"text": "no state"},
                   {"state": "x", "questions": "nope"})
    result, sink, dead = _run(router, source)
    assert result == {"decided": 2, "dead_letter": 3} and len(source.acked) == 5
    assert sink.sent[0]["ref"] == "cal-1" and sink.sent[0]["answers"]["event_type"]["answer"] == "interview" and sink.sent[0]["id"]
    assert router.log.find("cal-1") == sink.sent[0]["id"]
    errors = [d["error"] for d in dead.sent]
    assert errors[0].startswith("JSONDecodeError") and "must be a JSON object" in errors[1] and "no question set 'nope'" in errors[2]
    assert dead.sent[0]["message"] == "not json"


def test_transient_failures_are_retried_then_dead_lettered(router, monkeypatch):
    calls = []
    decide = router.decide

    def flaky(*args):
        calls.append(args[0])
        if args[0] == "always" or calls.count(args[0]) < 3:
            raise ConnectionError("db down")
        return decide(*args)

    monkeypatch.setattr(router, "decide", flaky)
    result, sink, dead = _run(router, Queue({"state": "flaky"}, {"state": "always"}), retries=2)
    assert result == {"decided": 1, "dead_letter": 1} and calls.count("flaky") == 3 and calls.count("always") == 3
    assert dead.sent[0]["error"] == "ConnectionError: db down"


def test_a_failing_sink_leaves_the_batch_unacknowledged(router):
    class Down(Queue):
        def send(self, messages):
            raise ConnectionError("sink down")

    source = Queue({"state": "standup"})
    with pytest.raises(ConnectionError):
        _run(router, source, sink=Down(), retries=1)
    assert source.acked == []


def test_redis_streams_read_a_group_and_reread_what_was_not_acknowledged(monkeypatch):
    fakeredis = pytest.importorskip("fakeredis")
    import redis

    server = fakeredis.FakeServer()
    monkeypatch.setattr(redis.Redis, "from_url", classmethod(lambda cls, url: fakeredis.FakeRedis(server=server)))
    out = worker.endpoint("redis://localhost:6379/0?stream=tickets", sink=True)
    out.send([b'{"state": "a"}', b'{"state": "b"}'])
    source = worker.endpoint("redis://localhost:6379/0?stream=tickets&group=g")
    first = source.receive(10)
    assert [raw for _, raw in first] == [b'{"state": "a"}', b'{"state": "b"}']
    source.ack([first[0][0]])
    again = worker.endpoint("redis://localhost:6379/0?stream=tickets&group=g")
    assert [raw for _, raw in again.receive(10)] == [b'{"state": "b"}']
    with pytest.raises(SystemExit, match="stream="):
        worker.endpoint("redis://localhost:6379/0")

    def interrupted(*args, **kwargs):
        raise redis.TimeoutError("Timeout reading from socket")  # what a stop signal during the blocking read gives

    monkeypatch.setattr(again.client, "xreadgroup", interrupted)
    assert again.receive(10) == []


def test_sqs_queues(monkeypatch):
    moto = pytest.importorskip("moto")
    import boto3

    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    with moto.mock_aws():
        boto3.client("sqs").create_queue(QueueName="tickets")
        queue = worker.endpoint("sqs://tickets")
        queue.send([json.dumps({"state": str(i)}).encode() for i in range(12)])
        got = []
        while len(got) < 12 and (batch := queue.receive(10)):
            got += batch
            queue.ack([handle for handle, _ in batch])
        assert sorted(int(json.loads(raw)["state"]) for _, raw in got) == list(range(12))
        assert queue.receive(10) == []


def test_endpoints_name_what_they_accept():
    assert isinstance(worker.endpoint("-", sink=True), worker.Stream) and isinstance(worker.endpoint("https://hooks.test/x", sink=True), worker.Webhook)
    with pytest.raises(SystemExit, match="not a source"):
        worker.endpoint("https://hooks.test/x")
