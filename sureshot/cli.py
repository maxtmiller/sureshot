"""The `sureshot` command: serve the API, decide, and run the feedback loop from a shell, cron or CI. It is configured
from the same environment variables as the API (see config.py), reads a .env in the current directory when
python-dotenv is installed, and prints JSON."""
import argparse
import json
import os
import sys
from pathlib import Path
from sureshot.calibrate import calibrate, prune, recheck
from sureshot.config import router_from_env
from sureshot.review import REASONS, review_queue
from sureshot.stats import stats
from sureshot.store import Store


def _store(args):
    path = args.db or os.environ.get("SURESHOT_DB")
    if not path:
        sys.exit("sureshot: set SURESHOT_DB or pass --db")
    if not Path(path).exists():
        sys.exit(f"sureshot: no log at {path}")
    return Store(path)


def _router(args):
    if args.db:
        os.environ["SURESHOT_DB"] = args.db
    if args.questions:
        os.environ["SURESHOT_QUESTIONS"] = args.questions
    return router_from_env()


def _json(text, name):
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        sys.exit(f"sureshot: {name} is not valid JSON ({e})")


def _costs(text):
    costs = _json(text, "--costs")
    if not isinstance(costs, dict) or "system2" not in costs:
        sys.exit('sureshot: --costs must be JSON with a "system2" cost per question, e.g. {"tier 0": 0, "system2": 0.004}')
    return costs


def serve(args):
    import uvicorn

    uvicorn.run("sureshot.api:app", host=args.host, port=args.port)


def decide(args):
    body = _json(Path(args.file).read_text() if args.file else sys.stdin.read(), args.file or "stdin")
    try:
        return _router(args).decide(body["state"], body.get("questions"), body.get("images", ()), body.get("thresholds"), body.get("escalate", True))
    except KeyError as e:
        sys.exit(f"sureshot: the request needs {e}")
    except ValueError as e:
        sys.exit(f"sureshot: {e}")


def retrain_local(args):
    from sureshot.learn import retrain
    from sureshot.local import LocalDecider

    path = args.model or os.environ.get("SURESHOT_LOCAL_MODEL")
    if not path:
        sys.exit("sureshot: set SURESHOT_LOCAL_MODEL or pass --model")
    current = LocalDecider.load(path) if Path(path).exists() else None
    result = retrain(_store(args), current, args.tier, args.target, args.min_samples, sample=args.sample, confident=args.confident)
    if result["decider"] is not current:
        result["decider"].save(path)
    return result["report"]


def test(args):
    try:
        report = _router(args).test(args.golden, args.set)
    except ValueError as e:
        sys.exit(f"sureshot: {e}")
    if args.min_accuracy is not None and (report["overall"] is None or report["overall"]["accuracy"] < args.min_accuracy):
        print(json.dumps(report, indent=1))
        sys.exit(1)
    return report


COMMANDS = {
    "serve": serve,
    "decide": decide,
    "calibrate": lambda a: calibrate(_store(a), a.target, a.min_samples, tier=a.tier, sample=a.sample, confident=a.confident),
    "retrain": retrain_local,
    "recheck": lambda a: recheck(_store(a), a.tier, a.sample, a.target, a.confident),
    "prune": lambda a: prune(_store(a), _costs(a.costs), a.tier, a.min_answers),
    "stats": lambda a: stats(_store(a), a.since, a.min_samples, costs=_costs(a.costs) if a.costs else None, sample=a.sample, confident=a.confident),
    "review": lambda a: review_queue(_store(a), a.limit, (a.reason,) if a.reason else REASONS, sample=a.sample, confident=a.confident),
    "test": test,
}


def parser():
    p = argparse.ArgumentParser(prog="sureshot", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", help="the decision log (default: SURESHOT_DB)")
    routed = argparse.ArgumentParser(add_help=False)
    routed.add_argument("--questions", help="a JSON or YAML questions file (default: SURESHOT_QUESTIONS)")
    sampled = argparse.ArgumentParser(add_help=False)
    sampled.add_argument("--sample", type=float, default=0.0, help="the review_queue sample rate your labels came from")
    sampled.add_argument("--confident", type=float, default=0.0, help="the review_queue confident rate your labels came from")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the HTTP API (needs sureshot[api])")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    s = sub.add_parser("decide", parents=[common, routed], help="decide one request, the JSON body of POST /decide")
    s.add_argument("file", nargs="?", help="the request (default: stdin)")

    s = sub.add_parser("calibrate", parents=[common, sampled], help="learn and save each question's threshold for a tier")
    s.add_argument("--target", type=float, default=0.95)
    s.add_argument("--min-samples", type=int, default=30)
    s.add_argument("--tier", type=int, default=0)

    s = sub.add_parser("retrain", parents=[common, sampled], help="train the LocalDecider on the log and save it where it helps (needs sureshot[local])")
    s.add_argument("--model", help="the saved LocalDecider, created if missing (default: SURESHOT_LOCAL_MODEL). A running API loads it on restart")
    s.add_argument("--target", type=float, default=0.95)
    s.add_argument("--min-samples", type=int, default=30)
    s.add_argument("--tier", type=int, default=0)

    s = sub.add_parser("recheck", parents=[common, sampled], help="test a tier's saved thresholds against your labels")
    s.add_argument("--tier", type=int, default=0)
    s.add_argument("--target", type=float, help="for thresholds with no calibrated target")
    s.set_defaults(sample=0.02)

    s = sub.add_parser("prune", parents=[common], help="stop asking a tier the questions it doesn't pay for")
    s.add_argument("--costs", required=True, help='cost per question of each stage as JSON, e.g. {"tier 0": 0, "system2": 0.004}')
    s.add_argument("--tier", type=int, default=0)
    s.add_argument("--min-answers", type=int, default=100)

    s = sub.add_parser("stats", parents=[common, sampled], help="usage, accuracy, cost, latency and drift report")
    s.add_argument("--since", type=float, default=0.0, help="unix time")
    s.add_argument("--min-samples", type=int, default=30)
    s.add_argument("--costs", help='cost per question of each stage as JSON, e.g. {"tier 0": 0, "system2": 0.004}')

    s = sub.add_parser("review", parents=[common, sampled], help="the questions to label next")
    s.add_argument("--limit", type=int, default=50)
    s.add_argument("--reason", choices=REASONS)

    s = sub.add_parser("test", parents=[common, routed], help="score the router on a golden JSONL file, without logging")
    s.add_argument("golden")
    s.add_argument("--set", help="the question set for cases without their own questions (default: the file's only set)")
    s.add_argument("--min-accuracy", type=float, help="exit with status 1 when overall accuracy is below this")
    return p


def main(argv=None):
    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass
    args = parser().parse_args(argv)
    result = COMMANDS[args.command](args)
    if result is not None:
        print(json.dumps(result, indent=1))
