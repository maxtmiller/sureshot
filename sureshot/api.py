"""HTTP API: POST /decide, /feedback and /calibrate with a bearer token (SURESHOT_API_KEY), and the OpenAI, Anthropic
and Gemini request formats for apps that already speak one (see formats.py), MCP at /mcp (see mcp.py), and SQL
functions for BigQuery and Snowflake at /warehouse (see warehouse.py). Run locally
with `uvicorn sureshot.api:app`."""
import os
import hmac
import json
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import parse_qs
from fastapi import BackgroundTasks, Body, FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field
from sureshot.adapters.base import post, post_stream
from sureshot.calibrate import calibrate
from sureshot.formats import (anthropic_request, anthropic_response, anthropic_stream, gemini_request, gemini_response, gemini_stream,
                              openai_request, openai_response, openai_stream, summary)
from sureshot.review import REASONS, review_queue
from sureshot.stats import stats
from sureshot.config import router_from_env
from sureshot.questions import load
from sureshot import mcp, slack, warehouse, webhooks

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
# A bad questions file fails at startup, not on the first request.
_sets = load(os.environ["SURESHOT_QUESTIONS"]) if os.environ.get("SURESHOT_QUESTIONS") else {}
_webhooks = webhooks.load(os.environ["SURESHOT_WEBHOOKS"]) if os.environ.get("SURESHOT_WEBHOOKS") else {}
_router = None


def _mcp_router():
    try:
        return router()
    except HTTPException as e:
        raise RuntimeError(e.detail) from e


_mcp = mcp.server(_mcp_router, _sets)


@asynccontextmanager
async def lifespan(app):
    # A session manager runs only once, so each start of the app makes a new one. Stateless, so any worker can answer
    # any request; the bearer token, not the Host header, is what keeps other sites out.
    _mcp.streamable_http_app(stateless_http=True, json_response=True, transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
    async with _mcp.session_manager.run():
        yield


app = FastAPI(title="sureshot", lifespan=lifespan)


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


@app.get("/questions")
def question_sets(authorization: str = Header(default="")):
    """The question sets from SURESHOT_QUESTIONS, by name: what no-code tools list in their dropdowns."""
    authorize(authorization)
    return {"sets": [{"name": name, "questions": questions} for name, questions in (router().questions or {}).items()]}


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



def _calls(rows, width):
    if not isinstance(rows, list) or not all(isinstance(r, list) and len(r) == width for r in rows):
        raise HTTPException(400, "each row must be the function's arguments: the question set's name and the text, e.g. sureshot('ticket_routing', body)")
    return [tuple(r[-2:]) for r in rows]


@app.post("/warehouse/bigquery")
def bigquery(body: dict = Body(...)):
    """A BigQuery remote function. BigQuery authenticates to Cloud Run with its own token, so the API key comes in
    the function's user_defined_context as api_key. Errors are an errorMessage; BigQuery retries a 503."""
    try:
        authorize(f"Bearer {(body.get('userDefinedContext') or {}).get('api_key', '')}")
        return {"replies": warehouse.decide(router(), _calls(body.get("calls"), 2))}
    except HTTPException as e:
        return JSONResponse({"errorMessage": str(e.detail)}, e.status_code)


@app.post("/warehouse/snowflake")
def snowflake(body: dict = Body(...), sf_custom_authorization: str = Header(default="")):
    """A Snowflake external function. Snowflake sends the function's HEADERS with an sf-custom- prefix, so the API
    key comes as HEADERS = ('authorization' = 'Bearer <key>')."""
    authorize(sf_custom_authorization)
    rows = body.get("data")
    results = warehouse.decide(router(), _calls(rows, 3))
    return {"data": [[row[0], result] for row, result in zip(rows, results)]}


class MCPEndpoint:
    """MCP's streamable HTTP transport, with the same bearer token as the rest of the API."""

    async def __call__(self, scope, receive, send):
        authorization = dict(scope["headers"]).get(b"authorization", b"").decode("latin-1")
        try:
            authorize(authorization)
        except HTTPException as e:
            return await JSONResponse({"detail": e.detail}, e.status_code)(scope, receive, send)
        await _mcp.session_manager.handle_request(scope, receive, send)


app.add_route("/mcp", MCPEndpoint(), methods=["GET", "POST", "DELETE"])


class Relay(StreamingResponse):
    """An upstream stream, closed however the response ends. When the client hangs up, Starlette stops iterating
    without closing the iterator, so a generator's finally would leave the upstream open, still generating."""

    def __init__(self, upstream):
        super().__init__(upstream.iter_bytes(), upstream.status_code, media_type=upstream.headers.get("content-type"))
        self.upstream = upstream

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.upstream.close()


def passthrough(fmt, body, path, headers, stream):
    """Forwards a request sureshot can't answer, unchanged, to SURESHOT_{fmt}_PASSTHROUGH_URL: the base URL you would
    give that API's SDK, with SURESHOT_{fmt}_PASSTHROUGH_API_KEY sent the way that API expects. A stream is relayed as
    it arrives."""
    url = os.environ.get(f"SURESHOT_{fmt}_PASSTHROUGH_URL")
    if not url:
        raise HTTPException(422, "sureshot answers requests for a JSON schema whose fields are all enums, booleans or integer ranges of 2 to 10 values; "
                                 f"set SURESHOT_{fmt}_PASSTHROUGH_URL to forward other requests")
    request = {"url": url.rstrip("/") + path, "json": body, "headers": headers(os.environ.get(f"SURESHOT_{fmt}_PASSTHROUGH_API_KEY")), "timeout": 300}
    if stream:
        return Relay(post_stream(**request))
    upstream = post(**request)
    return Response(upstream.content, upstream.status_code, media_type=upstream.headers.get("content-type"))


def answer(request, respond, stream=None, media_type="text/event-stream"):
    """Decides before responding, so a stream carries the whole answer and an error is still a 422."""
    try:
        result = router().decide(request["state"], request["questions"], request["images"])
    except ValueError as e:
        raise HTTPException(422, str(e))
    headers = {"x-sureshot": json.dumps(summary(result))}
    if stream:
        return Response(stream(respond(result)), media_type=media_type, headers=headers)
    return JSONResponse(respond(result), headers=headers)


@app.post("/v1/chat/completions")
def openai_chat(body: dict = Body(...), authorization: str = Header(default="")):
    """OpenAI's chat completions: point the client's base URL at <sureshot>/v1, with SURESHOT_API_KEY as its key."""
    authorize(authorization)
    if (request := openai_request(body)) is None:
        return passthrough("OPENAI", body, "/chat/completions", lambda key: {"Authorization": f"Bearer {key}"} if key else {}, body.get("stream"))
    include_usage = (body.get("stream_options") or {}).get("include_usage")
    return answer(request, lambda result: openai_response(body, request, result), body.get("stream") and (lambda c: openai_stream(c, include_usage)))


@app.post("/v1/messages")
def anthropic_messages(body: dict = Body(...), x_api_key: str = Header(default=""), authorization: str = Header(default=""),
                       anthropic_version: str = Header(default="2023-06-01"), anthropic_beta: str | None = Header(default=None)):
    """Anthropic's Messages: point the client's base URL at <sureshot>, with SURESHOT_API_KEY as its key."""
    authorize(f"Bearer {x_api_key}" if x_api_key else authorization)
    if (request := anthropic_request(body)) is None:
        version = {"anthropic-version": anthropic_version} | ({"anthropic-beta": anthropic_beta} if anthropic_beta else {})
        return passthrough("ANTHROPIC", body, "/v1/messages", lambda key: version | ({"x-api-key": key} if key else {}), body.get("stream"))
    return answer(request, lambda result: anthropic_response(body, request, result), body.get("stream") and anthropic_stream)


def gemini(version, model, body, key, method, stream=None, media_type=None):
    authorize(f"Bearer {key}")
    if (request := gemini_request(body)) is None:
        return passthrough("GEMINI", body, f"/{version}/models/{model}:{method}", lambda k: {"x-goog-api-key": k} if k else {}, stream)
    return answer(request, lambda result: gemini_response(model, request, result), stream, media_type)


@app.post("/{version}/models/{model}:generateContent")
def gemini_generate(version: str, model: str, body: dict = Body(...), x_goog_api_key: str = Header(default=""), key: str = ""):
    """Gemini's generateContent: point the client's base URL at <sureshot>, with SURESHOT_API_KEY as its key."""
    return gemini(version, model, body, x_goog_api_key or key, "generateContent")


@app.post("/{version}/models/{model}:streamGenerateContent")
def gemini_stream_generate(version: str, model: str, body: dict = Body(...), x_goog_api_key: str = Header(default=""), key: str = "", alt: str = ""):
    """Gemini's streamGenerateContent, as server-sent events with ?alt=sse or else a JSON array."""
    sse = alt == "sse"
    return gemini(version, model, body, x_goog_api_key or key, "streamGenerateContent?alt=sse" if sse else "streamGenerateContent", lambda r: gemini_stream(r, sse),
                  "text/event-stream" if sse else "application/json")
