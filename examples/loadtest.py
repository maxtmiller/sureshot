"""Load-test a running sureshot API: many clients at once, each sending /decide requests with a fresh text.

    python examples/loadtest.py http://localhost:8000 questions.json --clients 32 --requests 500

Reads SURESHOT_API_KEY (from the environment or .env). Every request uses a different text, so the response cache never
answers for the models. Prints throughput, latency percentiles and the errors by kind."""
import argparse
import collections
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
from dotenv import load_dotenv

parser = argparse.ArgumentParser()
parser.add_argument("url")
parser.add_argument("questions", help="a JSON file of questions, as in the README")
parser.add_argument("--clients", type=int, default=16)
parser.add_argument("--requests", type=int, default=200, help="in total")
parser.add_argument("--text", default="Hi, can you send me the Q3 numbers before our 3pm call today?")
args = parser.parse_args()
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
headers = {"Authorization": f"Bearer {os.environ['SURESHOT_API_KEY']}"}
questions = json.loads(Path(args.questions).read_text())
errors, latencies = collections.Counter(), []


def client(n):
    with httpx.Client(base_url=args.url, timeout=300) as c:
        for j in range(args.requests // args.clients):
            started = time.perf_counter()
            try:
                r = c.post("/decide", json={"state": f"{args.text} [{n}-{j}-{time.time_ns()}]", "questions": questions}, headers=headers)
                latencies.append((time.perf_counter() - started) * 1000)
                if r.status_code != 200:
                    errors[f"HTTP {r.status_code}: {r.text[:80]}"] += 1
            except httpx.HTTPError as e:
                errors[f"{type(e).__name__}: {str(e)[:80]}"] += 1


start = time.perf_counter()
with ThreadPoolExecutor(args.clients) as pool:
    list(pool.map(client, range(args.clients)))
wall = time.perf_counter() - start
latencies.sort()
done = len(latencies)
print(f"{done} requests in {wall:.1f}s with {args.clients} clients: {done / wall:.0f} requests/s, {sum(errors.values())} errors")
if latencies:
    print(f"latency ms: p50 {statistics.median(latencies):.0f}  p95 {latencies[int(.95 * done)]:.0f}  p99 {latencies[int(.99 * done)]:.0f}  max {latencies[-1]:.0f}")
for error, count in errors.most_common():
    print(f"  {count} x {error}")
