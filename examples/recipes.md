# Recipes

Short, copyable ways to plug sureshot into tools you already run. The question sets they use are in
[questions.yaml](questions.yaml).

- [Nightly upkeep](#nightly-upkeep): retrain, calibrate, recheck and prune on a schedule, with alerts
- [A CI gate for question changes](#a-ci-gate-for-question-changes)
- [LangChain and LlamaIndex](#langchain-and-llamaindex): no wrapper needed
- [n8n, Zapier and Make](#n8n-zapier-and-make): two HTTP calls
- [Email triage](#email-triage): the personal case
- [Support-ticket routing](#support-ticket-routing): the company case

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
reads unseen mail without marking it read and logs every decision, the dry run's included, to `triage.db`:
`sureshot review --db triage.db` lists what to label next, and `sureshot calibrate --db triage.db` learns thresholds
from the labels. Gmail needs IMAP turned on and an [app password](https://support.google.com/accounts/answer/185833).

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
