"""A queue worker: `sureshot worker --source URL --sink URL` reads messages, decides each, and writes the decisions on.

A message is the JSON body of POST /decide ({"state": ..., "questions": a set's name or inline, "ref": ...}); with
--set, messages may leave out questions. Each decision goes to the sink as the router's result plus the message's
ref. A fetched batch is decided in parallel, written to the sink, and only then acknowledged, so a crash means the
broker delivers it again (at least once: a row can be logged twice, and the answer cache keeps the models from being
asked twice). A message that can't be decided (not JSON, no state, a bad question) goes to the dead letter
destination at once; any other failure is retried with backoff first. The dead letter defaults to stderr, so nothing
is dropped unseen.

Sources and sinks, each needing its extra:
  sqs://<queue name>                                      sureshot[sqs]; region, credentials and AWS_ENDPOINT_URL as boto3 reads them
  redis://host:6379/0?stream=<name>&group=<group>         sureshot[redis]; a consumer group on a stream, field "data"
  kafka://host:9092[,host2:9092]/<topic>?group=<group>    sureshot[kafka]
Sinks only: an http(s):// webhook (each decision POSTed as JSON), "-" for stdout, "stderr"."""
import json
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlparse
from sureshot.adapters import base


def _need(module, extra):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError:
        raise SystemExit(f'sureshot: this source or sink needs {module}: pip install "sureshot[{extra}]"') from None


class SQS:
    def __init__(self, url):
        self.client = _need("boto3", "sqs").client("sqs")
        self.queue = self.client.get_queue_url(QueueName=url.netloc + url.path.rstrip("/"))["QueueUrl"]

    def receive(self, n):
        got = self.client.receive_message(QueueUrl=self.queue, MaxNumberOfMessages=min(n, 10), WaitTimeSeconds=5).get("Messages", [])
        return [(m["ReceiptHandle"], m["Body"].encode()) for m in got]

    def ack(self, handles):
        for i in range(0, len(handles), 10):
            self.client.delete_message_batch(QueueUrl=self.queue, Entries=[{"Id": str(j), "ReceiptHandle": h} for j, h in enumerate(handles[i:i + 10])])

    def send(self, messages):
        for i in range(0, len(messages), 10):
            failed = self.client.send_message_batch(QueueUrl=self.queue, Entries=[{"Id": str(j), "MessageBody": m.decode()} for j, m in enumerate(messages[i:i + 10])]).get("Failed")
            if failed:
                raise RuntimeError(f"SQS refused {len(failed)} messages: {failed[0].get('Message')}")


class RedisStream:
    def __init__(self, url):
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        if "stream" not in query:
            raise SystemExit("sureshot: a redis:// URL needs ?stream=<name> (and &group=<group> to read)")
        self.redis = _need("redis", "redis")
        self.client = self.redis.Redis.from_url(url._replace(query="").geturl())
        self.stream, self.group, self.consumer = query["stream"], query.get("group", "sureshot"), socket.gethostname()
        self.pending = True  # first re-read what this consumer took and never acknowledged

    def receive(self, n):
        if self.pending is True:
            try:
                self.client.xgroup_create(self.stream, self.group, id="0", mkstream=True)
            except Exception as e:
                if "BUSYGROUP" not in str(e):
                    raise
        try:
            got = self.client.xreadgroup(self.group, self.consumer, {self.stream: "0" if self.pending else ">"}, count=n, block=None if self.pending else 5000)
        except self.redis.TimeoutError:
            return []  # a stop signal interrupts the wait this way; the worker then stops

        entries = got[0][1] if got else []
        if self.pending and not entries:
            self.pending = False
            return self.receive(n)
        return [(entry_id, (fields or {}).get(b"data", b"")) for entry_id, fields in entries]  # a trimmed entry has no fields

    def ack(self, handles):
        self.client.xack(self.stream, self.group, *handles)

    def send(self, messages):
        with self.client.pipeline() as pipe:
            for m in messages:
                pipe.xadd(self.stream, {"data": m})
            pipe.execute()


class Kafka:
    def __init__(self, url):
        self.kafka, self.topic = _need("confluent_kafka", "kafka"), url.path.strip("/")
        self.config = {"bootstrap.servers": url.netloc}
        self.group = parse_qs(url.query).get("group", ["sureshot"])[0]
        self.consumer = self.producer = None

    def receive(self, n):
        if self.consumer is None:
            self.consumer = self.kafka.Consumer(self.config | {"group.id": self.group, "enable.auto.commit": False, "auto.offset.reset": "earliest"})
            self.consumer.subscribe([self.topic])
        got = []
        for m in self.consumer.consume(num_messages=n, timeout=5):
            if m.error():
                if m.error().fatal():
                    raise RuntimeError(f"Kafka: {m.error()}")
                continue
            got.append((m, m.value() or b""))
        return got

    def ack(self, handles):
        self.consumer.commit(asynchronous=False)  # the positions of everything consumed, all of it handled by now

    def send(self, messages):
        if self.producer is None:
            self.producer = self.kafka.Producer(self.config)
        errors = []
        for m in messages:
            self.producer.produce(self.topic, m, on_delivery=lambda err, _: err and errors.append(err))
        self.producer.flush()
        if errors:
            raise RuntimeError(f"Kafka refused {len(errors)} messages: {errors[0]}")


class Webhook:
    def __init__(self, url):
        self.url = url.geturl()

    def send(self, messages):
        for m in messages:
            base.ok(base.post(self.url, content=m, headers={"Content-Type": "application/json"}, timeout=30))


class Stream:
    def __init__(self, out):
        self.out = out

    def send(self, messages):
        for m in messages:
            self.out.write(m.decode() + "\n")
        self.out.flush()


BROKERS = {"sqs": SQS, "redis": RedisStream, "rediss": RedisStream, "kafka": Kafka}


def endpoint(url, sink=False):
    if sink and url in ("-", "stdout", "stderr"):
        return Stream(sys.stderr if url == "stderr" else sys.stdout)
    parsed = urlparse(url)
    if sink and parsed.scheme in ("http", "https"):
        return Webhook(parsed)
    if parsed.scheme not in BROKERS:
        kinds = "sqs://, redis://, kafka://" + (", http(s)://, - or stderr" if sink else "")
        raise SystemExit(f"sureshot: {url!r} is not a {'sink' if sink else 'source'}; use {kinds}")
    return BROKERS[parsed.scheme](parsed)


def _again(fn, retries, backoff):
    """fn(), tried again with backoff unless it fails with ValueError, which another try won't fix."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except ValueError:
            raise
        except Exception:
            if attempt == retries:
                raise
            time.sleep(backoff * 2 ** attempt)


def run(router, source, sink, dead_letter, questions=None, workers=8, retries=3, backoff=1.0, stop=lambda: False, say=lambda message: None):
    """Decides messages until stop() is true, finishing the batch in hand."""
    def one(raw):
        try:
            body = json.loads(raw)
            if not isinstance(body, dict) or not isinstance(body.get("state"), str):
                raise ValueError('a message must be a JSON object with "state", as the body of POST /decide')
            result = _again(lambda: router.decide(body["state"], body.get("questions", questions), body.get("images", ()), body.get("thresholds"),
                                                  body.get("escalate", True), body.get("ref")), retries, backoff)
            return True, json.dumps({"ref": body.get("ref")} | result).encode()
        except Exception as e:
            return False, json.dumps({"error": f"{type(e).__name__}: {e}", "message": raw.decode(errors="replace")}).encode()

    decided = failed = 0
    with ThreadPoolExecutor(workers) as pool:
        while not stop():
            batch = source.receive(workers * 4)
            if not batch:
                continue
            results = list(pool.map(one, [raw for _, raw in batch]))
            good, bad = [out for ok, out in results if ok], [out for ok, out in results if not ok]
            if good:
                _again(lambda: sink.send(good), retries, backoff)
            if bad:
                _again(lambda: dead_letter.send(bad), retries, backoff)
            source.ack([handle for handle, _ in batch])
            decided, failed = decided + len(good), failed + len(bad)
            say(f"{decided} decided, {failed} to the dead letter")
    return {"decided": decided, "dead_letter": failed}
