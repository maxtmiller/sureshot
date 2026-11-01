"""Route support tickets through a sureshot API, the company case, in the two calls a helpdesk integration makes:
route() when a ticket arrives, and reassigned() when an agent moves it to another team, which labels the decision so
calibration learns from the correction. Any language works the same way; this uses httpx.

Start the API with the ticket_routing questions, e.g. SURESHOT_QUESTIONS=examples/questions.yaml (or mount the file in
the Docker image), then:

    SURESHOT_URL=http://localhost:8000 SURESHOT_API_KEY=... python examples/ticket_routing.py "I was charged twice"
"""
import os
import sys
import httpx

URL, HEADERS = os.environ.get("SURESHOT_URL", "http://localhost:8000"), {"Authorization": f"Bearer {os.environ.get('SURESHOT_API_KEY', '')}"}
PRIORITIES = ["low", "normal", "high", "urgent"]


def route(text):
    """{"team", "priority", "decision"}: keep "decision" with the ticket, to report a reassignment."""
    response = httpx.post(f"{URL}/decide", json={"state": text, "questions": "ticket_routing"}, headers=HEADERS, timeout=60)
    response.raise_for_status()
    result = response.json()
    return {"team": result["answers"]["team"]["answer"], "priority": PRIORITIES[result["answers"]["priority"]["answer"]], "decision": result["id"]}


def reassigned(decision, team):
    """An agent moved the ticket to `team`: that is the right answer, so label the decision with it."""
    httpx.post(f"{URL}/feedback", json={"id": decision, "labels": {"team": team}}, headers=HEADERS, timeout=30).raise_for_status()


if __name__ == "__main__":
    print(route(" ".join(sys.argv[1:]) or sys.stdin.read()))
