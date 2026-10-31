"""Records real OpenAI exchanges into tests/recordings/ for test_recorded.py to replay. Saves the url, request body and
response only (never headers, so no key). Rerun when an adapter's request changes:

    OPENAI_API_KEY=... python tests/record_responses.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sureshot.adapters.base as backends
from sureshot import ChatDecider, ChatReasoner
from test_router import QUESTIONS

OUT = Path(__file__).parent / "recordings"
STATE = "Quarterly planning meeting with the finance team on Thursday at 10am. Please bring the budget draft."
real_post = backends.post


def record(name, run):
    exchanges = []

    def post(url, json, headers, timeout):
        response = real_post(url, json=json, headers=headers, timeout=timeout)
        exchanges.append({"url": url, "request": json, "status": response.status_code, "response": response.json()})
        return response

    backends.post = post
    try:
        run()
    except Exception as e:
        print(name, "raised", repr(e)[:120])
    finally:
        backends.post = real_post
    (OUT / f"{name}.json").write_text(json.dumps(exchanges, indent=1) + "\n")
    print(name, len(exchanges), "exchanges")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    record("openai_decider", lambda: ChatDecider("gpt-4o-mini", provider="openai").decide(STATE, QUESTIONS))
    record("openai_reasoner", lambda: ChatReasoner("gpt-4o-mini", provider="openai").answer(STATE, QUESTIONS))
    record("openai_reasoner_object", lambda: ChatReasoner("gpt-4o-mini", provider="openai", json_mode="object").answer(STATE, QUESTIONS))
    record("openai_deprecated_model", lambda: ChatDecider("gpt-4.1-nano", provider="openai").decide(STATE, {"needs_prep": QUESTIONS["needs_prep"]}))
