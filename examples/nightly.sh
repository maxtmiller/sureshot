#!/bin/sh
# Nightly upkeep for a sureshot deployment: retrain, calibrate, recheck, prune, then report what changed. Run it where
# the log is (the machine or container with SURESHOT_DB), with the same environment as the API:
#
#   0 3 * * *  /path/to/nightly.sh >> /var/log/sureshot-nightly.log 2>&1
#
# SAMPLE and CONFIDENT are the rates you label the review queue at. With SURESHOT_LOCAL_MODEL set, the classifier is
# tier 0 and is retrained, and the LLM is tier 1. Set COSTS (JSON, as for `sureshot prune`) to prune the LLM tier, and
# ALERT_WEBHOOK (a Slack-style incoming webhook) to be told when drift is flagged or a tier fails its labels.
set -eu
rates="--sample ${SAMPLE:-0.02} --confident ${CONFIDENT:-0.1}"
out=$(mktemp -d)
llm_tier=0
if [ -n "${SURESHOT_LOCAL_MODEL:-}" ]; then
  sureshot retrain $rates > "$out/retrain.json"
  llm_tier=1
fi
sureshot calibrate --tier "$llm_tier" $rates > "$out/calibrate.json"
sureshot recheck --tier "$llm_tier" $rates > "$out/recheck.json"
if [ -n "${COSTS:-}" ]; then
  sureshot prune --tier "$llm_tier" --costs "$COSTS" > "$out/prune.json"
fi
sureshot stats $rates > "$out/stats.json"

python3 - "$out" <<'EOF'
import json, os, sys, urllib.request

load = lambda name: json.load(open(os.path.join(sys.argv[1], name)))
problems = [f"tier {d['tier']} {d['key']}: drift in {', '.join(d['flags'])}" for d in load("stats.json")["drift"] if d["flags"]]
problems += [f"{key}: below target on labels, no longer trusted" for key, r in load("recheck.json").items() if r["failed"]]
print(f"{load('stats.json')['decisions']} decisions logged; " + ("; ".join(problems) or "nothing flagged"))
if problems and os.environ.get("ALERT_WEBHOOK"):
    message = json.dumps({"text": "sureshot nightly:\n" + "\n".join(problems)}).encode()
    urllib.request.urlopen(urllib.request.Request(os.environ["ALERT_WEBHOOK"], message, {"Content-Type": "application/json"}), timeout=10)
EOF
rm -r "$out"
