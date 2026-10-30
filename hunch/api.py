"""HTTP API: POST /decide, /feedback and /calibrate with a bearer token (HUNCH_API_KEY). Run locally with `uvicorn hunch.api:app`."""
import os
import hmac
import json
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field
from hunch.calibrate import calibrate
from hunch.review import REASONS, review_queue
from hunch.stats import stats
from hunch.config import router_from_env

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
app = FastAPI(title="hunch")
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
    questions: dict
    images: list[str] = []
    thresholds: dict[str, float] = {}
    escalate: bool = True


@app.get("/health")
def health():
    return {"ok": True}


def authorize(authorization):
    expected = os.environ.get("HUNCH_API_KEY")
    if not expected or not hmac.compare_digest(authorization, f"Bearer {expected}"):
        raise HTTPException(401, "missing or wrong bearer token")


def log():
    store = router().log
    if store is None:
        raise HTTPException(400, "set HUNCH_DB to log decisions")
    return store


class FeedbackRequest(BaseModel):
    id: str
    labels: dict


class CalibrateRequest(BaseModel):
    target_accuracy: float = 0.95
    min_samples: int = 30
    sample: float = Field(0.0, ge=0, le=1)


@app.post("/decide")
def decide(request: DecideRequest, authorization: str = Header(default="")):
    authorize(authorization)
    try:
        return router().decide(request.state, request.questions, request.images, request.thresholds, request.escalate)
    except ValueError as e:
        raise HTTPException(422, str(e))


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
                kept: float = Query(0.0, ge=0, le=1), authorization: str = Header(default="")):
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
    return stats(log(), since, min_samples, alpha, false_alarm_every, min_escalation_change, min_distribution_change, prices, sample, kept)


@app.get("/review")
def review(limit: int = 50, reason: str | None = None, sample: float = Query(0.0, ge=0, le=1), kept: float = Query(0.0, ge=0, le=1), authorization: str = Header(default="")):
    authorize(authorization)
    if reason is not None and reason not in REASONS:
        raise HTTPException(422, f"reason must be one of {list(REASONS)}")
    return review_queue(log(), limit, (reason,) if reason else REASONS, sample=sample, kept=kept)


@app.post("/calibrate")
def calibrate_thresholds(request: CalibrateRequest, authorization: str = Header(default="")):
    authorize(authorization)
    return calibrate(log(), request.target_accuracy, request.min_samples, sample=request.sample)
