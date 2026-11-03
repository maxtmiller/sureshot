# Recipes

Short, copyable ways to plug sureshot into tools you already run. The question sets they use are in
[questions.yaml](questions.yaml).

- [Nightly upkeep](#nightly-upkeep): retrain, calibrate, recheck and prune on a schedule, with alerts
- [Airflow, Dagster and Prefect](#airflow-dagster-and-prefect): the nightly loop and backfills as tasks
- [A CI gate for question changes](#a-ci-gate-for-question-changes)
- [LangChain and LlamaIndex](#langchain-and-llamaindex): no wrapper needed
- [n8n, Zapier and Make](#n8n-zapier-and-make): two HTTP calls
- [Email triage](#email-triage): the personal case
- [Support-ticket routing](#support-ticket-routing): the company case
- [Labels from webhooks](#labels-from-webhooks): Zendesk, GitHub, Jira, Linear or any automation, with no code
- [Labeling in Slack](#labeling-in-slack): the review queue as messages with a button per answer
- [Labeling in Label Studio](#labeling-in-label-studio): the review queue as tasks for an annotation team
- [Importing labeled data](#importing-labeled-data): start from the labels you already have
- [Backfills and batch jobs](#backfills-and-batch-jobs): a whole file at once, with System 2 at batch prices
- [SQL in BigQuery and Snowflake](#sql-in-bigquery-and-snowflake): `SELECT sureshot('ticket_routing', body) FROM tickets`
- [Claude Code, Claude Desktop and other agents](#claude-code-claude-desktop-and-other-agents): sureshot as MCP tools

## Nightly upkeep

sureshot only improves if calibration runs. [nightly.sh](nightly.sh) runs the whole loop with the
[command line](../README.md#command-line) and prints one line on what changed. Run it where the log is, with the same
environment as the API:

```bash
# crontab -e, on the machine with SURESHOT_DB
0 3 * * *  SAMPLE=0.02 CONFIDENT=0.1 ALERT_WEBHOOK=https://hooks.slack.com/... /path/to/nightly.sh >> /var/log/sureshot-nightly.log 2>&1
```

- `SAMPLE` and `CONFIDENT` are the rates you label the [review queue](../README.md#review-queue-what-to-label-next) at.
- With `SURESHOT_LOCAL_MODEL` set it retrains the classifier (tier 0) and calibrates the LLM as tier 1; without it the LLM is tier 0.
- `COSTS='{"tier 0": 0.0002, "system2": 0.004}'` also prunes questions the LLM tier doesn't pay for.
- `ALERT_WEBHOOK` (a Slack, Discord `/slack` or any Slack-style incoming webhook) gets a message when drift is flagged
  or a tier fails its labels. Without it, check the log.
- With `SURESHOT_SLACK_CHANNEL` set it posts the night's review items to [label in Slack](#labeling-in-slack), up to
  `SLACK_LIMIT` (30).

In Docker, copy the script into the volume and schedule `docker exec <container> sh /data/nightly.sh` from the host's
cron, so it runs next to the log with the API's environment.

## Airflow, Dagster and Prefect

Where a scheduler already runs, the loop can be tasks in it instead of cron. Each command-line step is a task whose
output is what the command prints, so later tasks can act on it: alert on drift from `stats`, or load a backfill's
output file. Options are the flags with underscores (`min_samples=50`), `True` for a bare flag, and a dict for JSON.
Workers need `pip install sureshot` and the API's environment (`SURESHOT_DB` as a `postgresql://` URL the API shares,
the model keys). A step that fails raises with the command line's message, so the task fails.

```python
# Airflow (2 or 3): results are XComs, options are templated
from sureshot.airflow import SureshotOperator

with DAG("sureshot_nightly", schedule="0 3 * * *"):
    rates = {"sample": 0.02, "confident": 0.1}
    calibrate = SureshotOperator(task_id="calibrate", command="calibrate", options={"tier": 1, **rates})
    recheck = SureshotOperator(task_id="recheck", command="recheck", options={"tier": 1, **rates})
    stats = SureshotOperator(task_id="stats", command="stats", options=rates)
    calibrate >> recheck >> stats

# Prefect
from sureshot.prefect import sureshot

@flow
def backfill(path: str):
    done = sureshot("decide", batch=path, set="ticket_routing")
    return sureshot("stats", wait_for=[done])

# Dagster: assets, or ops chained with start=
from sureshot.dagster import sureshot_asset, sureshot_op

tickets_decided = sureshot_asset("decide", name="tickets_decided", batch="tickets.csv", set="ticket_routing")
sureshot_stats = sureshot_asset("stats", deps=[tickets_decided])

calibrate, recheck = sureshot_op("calibrate", tier=1), sureshot_op("recheck", tier=1)

@job
def nightly():
    recheck(start=calibrate())
```

`decide` with `batch=` waits for System 2's batches, which can take hours, so give that task a long timeout; a retry
picks up where it stopped.

## A CI gate for question changes

Rewording a question can quietly cost accuracy. [golden-test.yml](golden-test.yml) is a GitHub Actions workflow for
your own repo: when a pull request touches `questions.yaml` or `golden.jsonl`, it scores the router on the golden set
and fails below 95%. Copy it to `.github/workflows/`, add `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` as secrets, and keep a
`golden.jsonl` of known answers, one `{"state": ..., "expected": {"needs_reply": true}}` per line. The same check runs
anywhere:

```bash
sureshot test golden.jsonl --questions questions.yaml --set email_triage --min-accuracy 0.95
```

It runs without a log, so it uses the default threshold (`SURESHOT_THRESHOLD`, 0.8). Point `--db` at a copy of the
production log to test with its calibrated thresholds.

## LangChain and LlamaIndex

Both already speak the OpenAI format, so pointing them at sureshot's [drop-in endpoint](../README.md#drop-in-for-openai-anthropic-and-gemini-apps)
is enough. Structured output then routes through sureshot:

```python
class Ticket(BaseModel):
    team: Literal["billing", "technical", "account", "sales"] = Field(description="Which team should handle it?")
    urgent: bool

# LangChain (langchain-openai): either method works
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", base_url="http://localhost:8000/v1", api_key=SURESHOT_API_KEY)
llm.with_structured_output(Ticket).invoke("I was charged twice")   # Ticket(team='billing', urgent=True)

# LlamaIndex (0.14 and later)
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-4o-mini", api_base="http://localhost:8000/v1", api_key=SURESHOT_API_KEY)
llm.structured_predict(Ticket, PromptTemplate("Route this ticket: {text}"), text="I was charged twice")
```

`ChatAnthropic(base_url="http://localhost:8000", ...)` works the same way through `/v1/messages`. Calls
that aren't structured output (chat, agents with optional tools) go to the `SURESHOT_*_PASSTHROUGH_URL` you set,
so one base URL serves the whole app. Older LlamaIndex versions send their schema as an optional tool
(`tool_choice: "auto"`), which sureshot passes through instead of answering.

## n8n, Zapier and Make

Any tool that can make an HTTP request can use sureshot: one call to decide, and optionally one to send back the right
answer when a person corrects it. Start the API with `SURESHOT_QUESTIONS` so the request only names a set.

| Step | Request |
| --- | --- |
| Decide | `POST https://<sureshot>/decide`, header `Authorization: Bearer <SURESHOT_API_KEY>`, JSON body `{"state": "<the text>", "questions": "ticket_routing"}` |
| Use the answer | `answers.team.answer`, `answers.priority.answer` (a level's position: 0 is Low), and `id` |
| Correct it | `POST https://<sureshot>/feedback`, same header, body `{"id": "<id from the decide step>", "labels": {"team": "technical"}}` |

- **n8n:** an *HTTP Request* node with *Send Body* as JSON; read the result with `{{ $json.answers.team.answer }}`.
- **Zapier:** *Webhooks by Zapier*, *Custom Request*, method POST, data as raw JSON.
- **Make:** the *HTTP* module's *Make a request*, body type *Raw*, content type JSON, with *Parse response* on.

Zapier and Make run in the cloud, so sureshot must be reachable over HTTPS from the internet (a host with a TLS proxy
in front of the Docker image, or a tunnel). Self-hosted n8n can reach it on the private network.

## Email triage

[email_triage.py](email_triage.py) triages an IMAP inbox: it flags mail that needs a reply today and moves promotions
out of the inbox, using the `email_triage` questions. It needs only the standard library and sureshot.

```bash
IMAP_HOST=imap.gmail.com IMAP_USER=you@gmail.com IMAP_PASSWORD=<an app password> \
SURESHOT_SYSTEM1_MODEL=gpt-4o-mini SURESHOT_SYSTEM1_PROVIDER=openai OPENAI_API_KEY=... \
python examples/email_triage.py --dry-run
```

`--dry-run` prints what it would do. Drop it once the choices look right, and run it from cron every 15 minutes. It
reads unseen mail without marking it read and logs every decision, the dry run's included, to `triage.db`. A few
minutes of labeling a week is what lets it learn your thresholds:

```bash
sureshot review --db triage.db --sample 0.1 --label    # shows each email and asks for the right answers
sureshot calibrate --db triage.db --sample 0.1         # learns thresholds from them
```
 Gmail needs IMAP turned on and an [app password](https://support.google.com/accounts/answer/185833).

## Support-ticket routing

[ticket_routing.py](ticket_routing.py) shows the two calls a helpdesk integration makes against the HTTP API:

```python
ticket = route("I was charged twice for my order")
# {"team": "billing", "priority": "urgent", "decision": "c982e8d4..."}   keep "decision" with the ticket

reassigned(ticket["decision"], "technical")   # an agent moved it: label the decision with the right team
```

Call `route()` from the helpdesk's new-ticket webhook and set the team and priority it returns. Call `reassigned()`
from its ticket-updated webhook when an agent changes the team. Every correction becomes a label, so the nightly
calibration learns from the agents' work with no separate labeling step. Sample the review queue too
(`SAMPLE=0.02`): corrections alone come only from tickets someone noticed were wrong.

## Labels from webhooks

With no code of your own, the helpdesk can send its own webhook to sureshot. Pass your id for the item as `ref` when
deciding, e.g. `route(text, ticket_id)` or `{"state": ..., "ref": "12345"}` to `POST /decide`, and point the tool's
webhook at `POST /feedback/webhook/<source>`, with a [webhooks file](webhooks.yaml) saying which event fields are which
labels:

```yaml
zendesk:                                 # POST /feedback/webhook/zendesk
  verify: zendesk                        # checks Zendesk's signature
  secret: ZENDESK_WEBHOOK_SECRET         # the environment variable holding it
  events:
    - ref: ticket_id                     # the ref given to /decide
      labels:
        team: {from: group, map: {Billing: billing, Technical Support: technical}}
```

```bash
SURESHOT_WEBHOOKS=examples/webhooks.yaml ZENDESK_WEBHOOK_SECRET=... sureshot serve
```

- `verify` is `zendesk`, `github`, `jira` or `linear` for their signed webhooks, or `bearer` (the default) for
  anything that can send `Authorization: Bearer $SURESHOT_API_KEY`, such as n8n, Zapier or Make. The API checks the
  file when it starts, including that each secret is set.
- An event takes the rules whose `when` fields match. A rule finds the decision by `ref` (the latest decision logged
  with it) or by `id` (a field holding the decision id itself, if the tool stores it). Each label comes `from` a field
  (dotted, `labels.0.name` for a list), translated by `map` when given, or is a fixed `value`.
- The response lists what was labeled and skipped (no logged decision, a value not in the map, an invalid answer), always
  with a 200, so the tool does not retry an event about an item sureshot never saw. A repeated event sets the same labels.
- Send an event that every item reaches, such as a ticket being solved with its final team, rather than only
  reassignments. Labels count toward confirming a threshold only on the decisions in the random review sample, and
  if only corrections arrive, those decisions get labels only when the model was wrong, so it looks worse than it is
  and its thresholds are never confirmed.
- Gmail has no webhook with the label in it (its push notifications say only that something changed), so for email
  triage label with `sureshot review --label`, or from a script that reads label changes and calls `/feedback`.

## Labeling in Slack

Reviewers who never open a terminal can label from a channel. `sureshot review --slack` posts the review queue there,
one message per decision: its text, each question with what the models said, and a button per answer (a menu past
five). A click saves the label and replaces the buttons with who answered.

1. Create the app from [slack-manifest.yaml](slack-manifest.yaml) with your sureshot URL (it needs the `chat:write`
   scope, and Interactivity pointed at `https://<sureshot>/slack/actions`), install it, and `/invite @sureshot` to the channel.
2. Give the API its Signing Secret (Basic Information > App Credentials; the Client Secret gets every click a 401)
   and the posting side its Bot User OAuth Token and channel id:

   ```bash
   SURESHOT_SLACK_SIGNING_SECRET=...  sureshot serve                  # receives the clicks
   SURESHOT_SLACK_BOT_TOKEN=xoxb-... SURESHOT_SLACK_CHANNEL=C0123 \
     sureshot review --slack --limit 30 --sample 0.02 --confident 0.1  # posts new items
   ```

3. Run the post from cron, or let [nightly.sh](nightly.sh) do it when `SURESHOT_SLACK_CHANNEL` is set.

- Each item is posted once: a later run posts only new ones, so unanswered messages stay where they are.
- Labels are saved as `slack:<user id>`. Anyone in the channel can answer, so pick the channel accordingly; a second
  click on another answer replaces the label.
- The text is posted as plain text, cut at 2,900 characters, so nothing in it can mention people. Images are not posted.
- Questions with more than 100 options don't fit a Slack menu; they are counted as `unanswerable` and left for
  `sureshot review --label`.

## Labeling in Label Studio

A team that labels in [Label Studio](https://labelstud.io) can work through the review queue there.
`sureshot label-studio push` sends new queue items of one question set to a project named `sureshot: <set>`, created
on first use with a labeling interface built from the questions (one single choice per question), one task per
decision. `sureshot label-studio pull` saves the project's annotations as labels.

```bash
export SURESHOT_LABEL_STUDIO_URL=http://localhost:8080 SURESHOT_LABEL_STUDIO_TOKEN=...   # Account & Settings > Personal Access Token
sureshot label-studio push --questions examples/questions.yaml --set ticket_routing --limit 100 --sample 0.02 --confident 0.1
sureshot label-studio pull --questions examples/questions.yaml --set ticket_routing
```

- Run both from cron, e.g. after [nightly.sh](nightly.sh): pull first, so the night's calibration sees the day's labels.
- Each item is sent once, to Slack or Label Studio. A task shows every question of its decision; answering the ones
  that weren't queued labels them too.
- Where several annotators answered a question, it is labeled only when they agree (`skipped` counts the rest). A
  pull labels only what changed, so pulling every night doesn't reorder the labels confirmation reads.
- Changing a question set updates the project's interface on the next push. Label Studio refuses to drop a question
  that already has annotations; make a new set (and so a new project) for that.
- Label Studio 1.23 turns legacy tokens off by default; a personal access token works either way.

## Importing labeled data

Labeled examples you already have, from an old classifier, a spreadsheet or a dataset, give calibration and
`retrain` something to learn from before any new labels arrive. Put them in a CSV (with a header) or JSONL file
with the text and a column per question:

```csv
text,team,priority
I was charged twice for my order,billing,High
The app crashes when I upload a photo,technical,2
```

```bash
sureshot import labels.csv --questions examples/questions.yaml --set ticket_routing --limit 2000
sureshot calibrate --target 0.9          # grades System 1 on the imported labels
sureshot retrain                         # trains the classifier on them (with SURESHOT_LOCAL_MODEL)
```

- Each row is asked of System 1, one call per question, so `--limit` caps the cost. `--escalate` also asks System 2
  about what System 1 is unsure of, which costs more and isn't needed: the labels are the truth.
- Cells are read as a spreadsheet writes them: yes/no, true/false or 1/0; a score level's position or name; a
  choice's key in any case. An empty cell is no label. Rows with an invalid label are skipped and listed in `errors`.
- Running it again skips rows already imported (same text), so a stopped import can be resumed.
- Imported decisions count for grading and training, but not as today's traffic: they are never in the random review
  sample, so their labels can't confirm a threshold (old data need not look like what comes in now), and `stats`,
  drift, `prune` and the review queue leave them out. Confirmation still comes from labels on live decisions.
- A Hugging Face dataset can be saved as JSONL first, mapping class numbers to your option keys:
  `datasets.load_dataset("banking77")["train"].to_json("labels.jsonl")`.

## Backfills and batch jobs

A backlog, or a nightly export, can be decided in one command. Every row is decided as a live request would be and
written back with its answers:

```bash
sureshot decide --batch tickets.csv --questions examples/questions.yaml --set ticket_routing
# {"rows": 5000, "decided": 5000, "errors": 0, "escalated": 640, "out": "tickets.decided.csv"}
```

- Each row gets `sureshot_id` (the decision id, for `/feedback`), and per question `sureshot_<key>`,
  `sureshot_<key>_source` (`tier 0`, `system2`...) and `sureshot_<key>_confidence`. Rows without text get
  `sureshot_error`. CSV, JSONL and Parquet (`pip install "sureshot[parquet]"`) work, in and out; `--out` picks the output file.
- System 1 answers `--workers` rows at once. When System 2 is OpenAI (`SURESHOT_SYSTEM2_PROVIDER=openai`) or Claude
  through the API, what it must answer goes out as batches at half the price, checked every `--poll` seconds; most
  finish within an hour, and the limit is 24. Other System 2s are asked directly.
- Stopping is safe: progress is kept in `<out>.progress`, so the same command picks up where it was without asking
  or logging anything twice, and waits for batches already sent. Questions a batch could not answer (or Claude
  declined, since batches take no fallback model) are asked directly.
- Unlike `sureshot import`, these are live decisions: they are in the random review sample and in `stats`, so a
  backfill's labels can confirm thresholds.

## SQL in BigQuery and Snowflake

Analysts can decide rows where they already are: a BigQuery remote function or a Snowflake external function sends
batches of `(question set, text)` to a running sureshot, and each row gets back a JSON object with the same fields
`decide --batch` writes (`sureshot_id`, `sureshot_<key>`, `_source`, `_confidence`, or `sureshot_error` for a row
with no text or an unknown set). System 2 is asked directly, so a batch takes as long as its slowest escalation.

Both run the published image on Cloud Run, with a shared log in Postgres and the questions file as a secret. Cloud
Run pulls from Artifact Registry, so copy the image there first:

```bash
IMAGE=us-central1-docker.pkg.dev/$PROJECT/sureshot/sureshot:latest
docker pull --platform linux/amd64 ghcr.io/maxtmiller/sureshot:latest
docker tag ghcr.io/maxtmiller/sureshot:latest $IMAGE && docker push $IMAGE
gcloud run deploy sureshot --image $IMAGE --region us-central1 --no-allow-unauthenticated --timeout 900 \
  --set-env-vars SURESHOT_DB=postgresql://...,SURESHOT_CACHE=postgresql://...,SURESHOT_QUESTIONS=/config/questions.yaml,SURESHOT_SYSTEM1_MODEL=gpt-4o-mini,SURESHOT_SYSTEM1_PROVIDER=openai \
  --set-secrets SURESHOT_API_KEY=sureshot-api-key:latest,OPENAI_API_KEY=openai-api-key:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest,/config/questions.yaml=sureshot-questions:latest
```

Warehouses retry a batch that failed or timed out, so a row can be decided and logged twice; `SURESHOT_CACHE` keeps
the retry from asking the models again.

**BigQuery.** A connection's service account calls Cloud Run; the [remote functions guide](https://cloud.google.com/bigquery/docs/remote-functions)
has the details.

```bash
bq mk --connection --location=US --connection_type=CLOUD_RESOURCE sureshot
bq show --connection $PROJECT.us.sureshot    # its serviceAccountId
gcloud run services add-iam-policy-binding sureshot --region us-central1 --member serviceAccount:<serviceAccountId> --role roles/run.invoker
```

```sql
CREATE FUNCTION analytics.sureshot(question_set STRING, text STRING) RETURNS JSON
REMOTE WITH CONNECTION `my-project.us.sureshot`
OPTIONS (endpoint = 'https://sureshot-<hash>.a.run.app/warehouse/bigquery',
         user_defined_context = [("api_key", "<SURESHOT_API_KEY>")], max_batching_rows = 50);

SELECT id, JSON_VALUE(r, '$.sureshot_team') AS team, JSON_VALUE(r, '$.sureshot_id') AS decision
FROM (SELECT id, analytics.sureshot('ticket_routing', body) AS r FROM support.tickets);
```

BigQuery authenticates to Cloud Run with its own token, so sureshot's key travels in `user_defined_context`, where
anyone who can read the function's definition can see it; Cloud Run's invoker role is the gate that matters.

**Snowflake.** External functions reach a service through an API gateway; with Cloud Run, that is Google Cloud API
Gateway, set up as in Snowflake's [guide for Google Cloud](https://docs.snowflake.com/en/sql-reference/external-functions-creating-gcp),
with the gateway's backend `https://sureshot-<hash>.a.run.app/warehouse/snowflake` and its service account given
`roles/run.invoker`. Snowflake sends a function's `HEADERS` with an `sf-custom-` prefix, which is where the key goes:

```sql
CREATE EXTERNAL FUNCTION sureshot(question_set VARCHAR, text VARCHAR) RETURNS VARIANT
API_INTEGRATION = sureshot_gateway
HEADERS = ('authorization' = 'Bearer <SURESHOT_API_KEY>')
MAX_BATCH_ROWS = 50
AS 'https://<gateway-host>/warehouse/snowflake';

SELECT id, r:sureshot_team::string AS team, r:sureshot_id::string AS decision
FROM (SELECT id, sureshot('ticket_routing', body) AS r FROM support.tickets);
```

Make the function `SECURE` to hide the header from everyone but its owner. Label the decisions as usual, with the
`sureshot_id` each row returned (`POST /feedback`, or a [webhook](#labels-from-webhooks) from where the outcome lands).

## Claude Code, Claude Desktop and other agents

`sureshot mcp` (`pip install "sureshot[mcp]"`) gives an agent four tools: `decide`, `feedback`, `review` and `stats`.
It reads the same environment as the command line, and each question set in the questions file can be passed to
`decide` by name and read as `sureshot://questions/<set>`.

```bash
# Claude Code, run where the log is
claude mcp add sureshot -e SURESHOT_DB=$PWD/sureshot.db -e SURESHOT_QUESTIONS=$PWD/examples/questions.yaml -- sureshot mcp

# or a running `sureshot serve`, which answers MCP at /mcp with its API key
claude mcp add --transport http sureshot https://<sureshot>/mcp --header "Authorization: Bearer $SURESHOT_API_KEY"
```

Claude Desktop starts it from `claude_desktop_config.json` (Settings > Developer > Edit Config), with the full path
`which sureshot` prints and the model keys it needs in `env`:

```json
{"mcpServers": {"sureshot": {"command": "/path/to/sureshot", "args": ["mcp"],
  "env": {"SURESHOT_DB": "/path/to/sureshot.db", "SURESHOT_QUESTIONS": "/path/to/questions.yaml", "OPENAI_API_KEY": "..."}}}}
```

- Ask the agent to "work through the sureshot review queue with me": it calls `review`, asks you each question, and
  saves your answers with `feedback`. Give it the rates you label at, e.g. "with sample 0.02 and confident 0.1".
- Labels are saved as `mcp`. Calibration treats them as the truth, so they must be your answers, not the agent's:
  the tools say so, and `review` leaves out what the models answered on randomly sampled items, so neither you nor
  the agent is steered by them.
