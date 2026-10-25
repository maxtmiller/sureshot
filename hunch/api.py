"""HTTP API: POST /decide, /feedback and /calibrate with a bearer token (HUNCH_API_KEY). Run locally with `uvicorn hunch.api:app`."""
import os
import hmac
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from hunch.calibrate import calibrate
from hunch.config import router_from_env

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
app = FastAPI(title="hunch")
_router = None


def router():
    global _router
    if _router is None:
        built = router_from_env()
        if hasattr(built.system1, "check"):
            try:
                built.system1.check()
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


@app.post("/calibrate")
def calibrate_thresholds(request: CalibrateRequest, authorization: str = Header(default="")):
    authorize(authorization)
    return calibrate(log(), request.target_accuracy, request.min_samples)
