# Recipes

Short, copyable ways to plug sureshot into tools you already run. The question sets they use are in
[questions.yaml](questions.yaml).

- [Nightly upkeep](#nightly-upkeep): retrain, calibrate, recheck and prune on a schedule, with alerts
- [A CI gate for question changes](#a-ci-gate-for-question-changes)
- [LangChain and LlamaIndex](#langchain-and-llamaindex): no wrapper needed
- [n8n, Zapier and Make](#n8n-zapier-and-make): two HTTP calls
- [Email triage](#email-triage): the personal case
- [Support-ticket routing](#support-ticket-routing): the company case
- [Labels from webhooks](#labels-from-webhooks): Zendesk, GitHub, Jira, Linear or any automation, with no code
- [Labeling in Slack](#labeling-in-slack): the review queue as messages with a button per answer

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
2. Give the API its signing secret and the posting side its token and channel:

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
