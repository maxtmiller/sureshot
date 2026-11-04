"""Alerts to a Slack incoming webhook or any URL (SURESHOT_ALERT_URL) when a tier is demoted, a recheck fails or stats
finds drift. A Slack URL gets {"text"}; any other gets {"event", "text", "items"}. The commands that calibrate, recheck,
retrain or report send them (the CLI, and POST /calibrate); the library functions never do."""
import os
import sys
import httpx


def send(event, text, items):
    """Posts an alert when SURESHOT_ALERT_URL is set. An alert that can't be delivered is reported on stderr, so it never
    fails the job that raised it."""
    url = os.environ.get("SURESHOT_ALERT_URL")
    if not url or not items:
        return
    body = {"text": text} if url.startswith("https://hooks.slack.com/") else {"event": event, "text": text, "items": items}
    try:
        httpx.post(url, json=body, timeout=10).raise_for_status()
    except httpx.HTTPError as e:
        print(f"sureshot: could not send the {event} alert to SURESHOT_ALERT_URL: {e}", file=sys.stderr)


def demoted(before, after, tier, event="demotion"):
    """Alerts on the questions whose threshold for a tier went from a number to infinity (the tier is no longer
    trusted with them) between two store.thresholds(tier) readings."""
    keys = sorted(key for key, value in after.items() if value == float("inf") and before.get(key, float("inf")) != float("inf"))
    what = "failed recheck against your labels" if event == "recheck" else "were demoted"
    send(event, f"sureshot: on tier {tier}, {', '.join(keys)} {what}; System 2 answers them until a calibration certifies a threshold again",
         [{"tier": tier, "key": key, "threshold": before[key]} for key in keys])


def drift(report):
    """Alerts on the questions stats flagged as drifting."""
    items = [{k: d[k] for k in ("tier", "key", "flags", "change")} for d in report.get("drift", []) if d["flags"]]
    send("drift", "sureshot: drift on " + "; ".join(f"tier {d['tier']} {d['key']} ({', '.join(d['flags'])})" for d in items), items)
