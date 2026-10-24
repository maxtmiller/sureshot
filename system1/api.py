"""HTTP API: POST /decide with a bearer token (SYSTEM1_API_KEY). Run locally with `uvicorn system1.api:app`."""
import os
import hmac
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from system1.backends import Decider, default_system2
from system1.router import Router

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
app = FastAPI(title="system1")
_router = None


def router():
    global _router
    if _router is None:
        _router = Router(Decider(), default_system2(), float(os.environ.get("SYSTEM1_THRESHOLD", 0.8)))
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


@app.post("/decide")
def decide(request: DecideRequest, authorization: str = Header(default="")):
    expected = os.environ.get("SYSTEM1_API_KEY")
    if not expected or not hmac.compare_digest(authorization, f"Bearer {expected}"):
        raise HTTPException(401, "missing or wrong bearer token")
    try:
        return router().decide(request.state, request.questions, request.images, request.thresholds, request.escalate)
    except ValueError as e:
        raise HTTPException(422, str(e))
