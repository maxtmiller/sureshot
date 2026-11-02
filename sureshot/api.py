"""HTTP API: POST /decide, /feedback and /calibrate with a bearer token (SURESHOT_API_KEY), and the OpenAI, Anthropic
and Gemini request formats for apps that already speak one (see formats.py). Run locally with `uvicorn sureshot.api:app`."""
import os
import hmac
import json
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import parse_qs
from fastapi import BackgroundTasks, Body, FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from sureshot.adapters.base import post
from sureshot.calibrate import calibrate
from sureshot.formats import anthropic_request, anthropic_response, gemini_request, gemini_response, openai_request, openai_response, summary
from sureshot.review import REASONS, review_queue
from sureshot.stats import stats
from sureshot.config import router_from_env
from sureshot.questions import load
from sureshot import slack, webhooks

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
if os.environ.get("SURESHOT_QUESTIONS"):
    load(os.environ["SURESHOT_QUESTIONS"])  # a bad questions file fails at startup, not on the first request
_webhooks = webhooks.load(os.environ["SURESHOT_WEBHOOKS"]) if os.environ.get("SURESHOT_WEBHOOKS") else {}
app = FastAPI(title="sureshot")
_router = None


def router():
    global _router
    if _router is None:
        built = router_from_env()
        for tier in built.tiers:
            if hasattr(tier, "check"):
                try:
                    tier.check()
                except RuntimeError as e:
                    raise HTTPException(503, f"System 1 is not ready: {e}")
        _router = built
    return _router


class DecideRequest(BaseModel):
    state: str
    questions: dict | str | None = None
    images: list[str] = []
    thresholds: dict[str, float] = {}
    escalate: bool = True
    ref: str | None = None


@app.get("/health")
def health():
    return {"ok": True}


def authorize(authorization):
    expected = os.environ.get("SURESHOT_API_KEY")
    if not expected or not hmac.compare_digest(authorization, f"Bearer {expected}"):
        raise HTTPException(401, "missing or wrong bearer token")


def log():
    store = router().log
    if store is None:
        raise HTTPException(400, "set SURESHOT_DB to log decisions")
    return store


class FeedbackRequest(BaseModel):
    id: str
    labels: dict


class CalibrateRequest(BaseModel):
    target_accuracy: float = 0.95
    min_samples: int = 30
    sample: float = Field(0.0, ge=0, le=1)
    confident: float = Field(0.0, ge=0, le=1)


@app.post("/decide")
def decide(request: DecideRequest, authorization: str = Header(default="")):
    authorize(authorization)
    try:
        return router().decide(request.state, request.questions, request.images, request.thresholds, request.escalate, request.ref)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/feedback/webhook/{source}")
async def feedback_webhook(source: str, request: Request):
    """Labels from another tool's webhook, by the rules SURESHOT_WEBHOOKS gives the source (see webhooks.py). It
    answers 200 with what was labeled and skipped, so the sender does not retry an event that matched nothing."""
    if source not in _webhooks:
        raise HTTPException(404, f"no webhook source {source!r} in SURESHOT_WEBHOOKS")
    body = await request.body()
    if not webhooks.verify(_webhooks[source], request.headers, body):
        raise HTTPException(401, "missing or wrong webhook signature")
    try:
        event = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(400, "the webhook body must be JSON")
    return {"events": await run_in_threadpool(lambda: webhooks.apply(log(), source, _webhooks[source], event))}


@app.post("/slack/actions")
async def slack_actions(request: Request, background: BackgroundTasks):
    """A click on a review message posted by `sureshot review --slack` (see slack.py). Slack wants a reply within 3
    seconds, so the label is saved before replying and the message is updated after."""
    if not os.environ.get("SURESHOT_SLACK_SIGNING_SECRET"):
        raise HTTPException(404, "set SURESHOT_SLACK_SIGNING_SECRET to label from Slack")
    body = await request.body()
    if not slack.verify(request.headers, body):
        raise HTTPException(401, "missing, wrong or stale Slack signature")
    try:
        payload = json.loads(parse_qs(body.decode())["payload"][0])
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(400, "expected Slack's form-encoded payload")
    reply = await run_in_threadpool(lambda: slack.handle(log(), payload))
    if reply is not None and payload.get("response_url"):
        background.add_task(post, payload["response_url"], json=reply, timeout=30)
    return Response(status_code=200)


@app.post("/feedback")
def feedback(request: FeedbackRequest, authorization: str = Header(default="")):
    authorize(authorization)
    store = log()
    try:
        for key, answer in request.labels.items():
            store.label(request.id, key, answer)
    except KeyError as e:
        raise HTTPException(404, str(e.args[0]))
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"ok": True}


@app.get("/stats")
def usage_stats(since: float = 0.0, min_samples: int = Query(30, ge=1), alpha: float = Query(0.01, gt=0, lt=1),
                false_alarm_every: float = Query(100_000, gt=1), min_escalation_change: float = Query(0.05, ge=0, le=1),
                min_distribution_change: float = Query(0.1, ge=0, le=1), costs: str | None = None, sample: float = Query(0.0, ge=0, le=1),
                confident: float = Query(0.0, ge=0, le=1), authorization: str = Header(default="")):
    """costs is JSON, e.g. {"tier 0": 0, "tier 1": 0.0002, "system2": 0.004}: a cost per question of each stage."""
    authorize(authorization)
    prices = None
    if costs is not None:
        try:
            prices = json.loads(costs)
        except json.JSONDecodeError:
            raise HTTPException(422, "costs must be JSON, e.g. {\"system2\": 0.004}")
        if not isinstance(prices, dict) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in prices.values()):
            raise HTTPException(422, "costs must map each stage to a number")
        if "system2" not in prices:
            raise HTTPException(422, 'costs needs a "system2" cost per question')
    return stats(log(), since, min_samples, alpha, false_alarm_every, min_escalation_change, min_distribution_change, prices, sample, confident)


@app.get("/review")
def review(limit: int = 50, reason: str | None = None, sample: float = Query(0.0, ge=0, le=1), confident: float = Query(0.0, ge=0, le=1), authorization: str = Header(default="")):
    authorize(authorization)
    if reason is not None and reason not in REASONS:
        raise HTTPException(422, f"reason must be one of {list(REASONS)}")
    return review_queue(log(), limit, (reason,) if reason else REASONS, sample=sample, confident=confident)


@app.post("/calibrate")
def calibrate_thresholds(request: CalibrateRequest, authorization: str = Header(default="")):
    authorize(authorization)
    return calibrate(log(), request.target_accuracy, request.min_samples, sample=request.sample, confident=request.confident)



def passthrough(fmt, body, path, headers):
    """Forwards a request sureshot can't answer, unchanged, to SURESHOT_{fmt}_PASSTHROUGH_URL: the base URL you would
    give that API's SDK, with SURESHOT_{fmt}_PASSTHROUGH_API_KEY sent the way that API expects."""
    url = os.environ.get(f"SURESHOT_{fmt}_PASSTHROUGH_URL")
    if not url or body.get("stream"):
        raise HTTPException(422, "sureshot answers requests for a JSON schema whose fields are all enums, booleans or integer ranges of 2 to 10 values, "
                                 f"without streaming; set SURESHOT_{fmt}_PASSTHROUGH_URL to forward other requests, except streaming ones")
    upstream = post(url.rstrip("/") + path, json=body, headers=headers(os.environ.get(f"SURESHOT_{fmt}_PASSTHROUGH_API_KEY")), timeout=300)
    return Response(upstream.content, upstream.status_code, media_type=upstream.headers.get("content-type"))


def answer(request, respond):
    try:
        result = router().decide(request["state"], request["questions"], request["images"])
    except ValueError as e:
        raise HTTPException(422, str(e))
    return JSONResponse(respond(result), headers={"x-sureshot": json.dumps(summary(result))})


@app.post("/v1/chat/completions")
def openai_chat(body: dict = Body(...), authorization: str = Header(default="")):
    """OpenAI's chat completions: point the client's base URL at <sureshot>/v1, with SURESHOT_API_KEY as its key."""
    authorize(authorization)
    if (request := openai_request(body)) is None:
        return passthrough("OPENAI", body, "/chat/completions", lambda key: {"Authorization": f"Bearer {key}"} if key else {})
    return answer(request, lambda result: openai_response(body, request, result))


@app.post("/v1/messages")
def anthropic_messages(body: dict = Body(...), x_api_key: str = Header(default=""), authorization: str = Header(default=""),
                       anthropic_version: str = Header(default="2023-06-01"), anthropic_beta: str | None = Header(default=None)):
    """Anthropic's Messages: point the client's base URL at <sureshot>, with SURESHOT_API_KEY as its key."""
    authorize(f"Bearer {x_api_key}" if x_api_key else authorization)
    if (request := anthropic_request(body)) is None:
        version = {"anthropic-version": anthropic_version} | ({"anthropic-beta": anthropic_beta} if anthropic_beta else {})
        return passthrough("ANTHROPIC", body, "/v1/messages", lambda key: version | ({"x-api-key": key} if key else {}))
    return answer(request, lambda result: anthropic_response(body, request, result))


@app.post("/{version}/models/{model}:generateContent")
def gemini_generate(version: str, model: str, body: dict = Body(...), x_goog_api_key: str = Header(default=""), key: str = ""):
    """Gemini's generateContent: point the client's base URL at <sureshot>, with SURESHOT_API_KEY as its key."""
    authorize(f"Bearer {x_goog_api_key or key}")
    if (request := gemini_request(body)) is None:
        return passthrough("GEMINI", body, f"/{version}/models/{model}:generateContent", lambda k: {"x-goog-api-key": k} if k else {})
    return answer(request, lambda result: gemini_response(model, request, result))
