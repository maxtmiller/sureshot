import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sureshot import Store
from sureshot.router import Router
from test_router import QUESTIONS, FakeClaude, FakeDecider


@pytest.fixture
def client(tmp_path, monkeypatch):
    import sureshot.api as api

    monkeypatch.setenv("SURESHOT_API_KEY", "secret")
    monkeypatch.setattr(api, "_router", Router(FakeDecider(), FakeClaude(), log=Store(tmp_path / "sureshot.db"), questions={"events": QUESTIONS}))
    return TestClient(api.app)


def test_bigquery_answers_each_call_and_reports_bad_rows_in_place(client):
    body = {"requestId": "r1", "caller": "job", "sessionUser": "a@b.c", "userDefinedContext": {"api_key": "secret"},
            "calls": [["events", "standup with the team"], ["events", None], ["nope", "standup"]]}
    response = client.post("/warehouse/bigquery", json=body, headers={"Authorization": "Bearer google-identity-token"})
    good, empty, unknown = response.json()["replies"]
    assert good["sureshot_event_type"] == "interview" and good["sureshot_event_type_source"] == "tier 0" and good["sureshot_id"]
    assert good["sureshot_importance"] == 4 and good["sureshot_importance_source"] == "system2"
    assert empty == {"sureshot_error": "no text"} and "no question set 'nope'" in unknown["sureshot_error"]


def test_bigquery_errors_are_an_error_message(client, monkeypatch):
    import sureshot.api as api

    calls = [["events", "standup"]]
    wrong = client.post("/warehouse/bigquery", json={"calls": calls, "userDefinedContext": {"api_key": "wrong"}})
    assert wrong.status_code == 401 and wrong.json() == {"errorMessage": "missing or wrong bearer token"}
    assert client.post("/warehouse/bigquery", json={"calls": calls}).status_code == 401
    shape = client.post("/warehouse/bigquery", json={"calls": [["standup"]], "userDefinedContext": {"api_key": "secret"}})
    assert shape.status_code == 400 and "question set's name and the text" in shape.json()["errorMessage"]

    def not_ready():
        raise HTTPException(503, "System 1 is not ready: loading")

    monkeypatch.setattr(api, "router", not_ready)
    retry = client.post("/warehouse/bigquery", json={"calls": calls, "userDefinedContext": {"api_key": "secret"}})
    assert retry.status_code == 503 and retry.json() == {"errorMessage": "System 1 is not ready: loading"}


def test_snowflake_answers_rows_by_number(client):
    body = {"data": [[0, "events", "standup with the team"], [1, "events", ""]]}
    rows = client.post("/warehouse/snowflake", json=body, headers={"sf-custom-authorization": "Bearer secret"}).json()["data"]
    assert [n for n, _ in rows] == [0, 1] and rows[0][1]["sureshot_event_type"] == "interview" and rows[1][1] == {"sureshot_error": "no text"}
    assert client.post("/warehouse/snowflake", json=body, headers={"Authorization": "Bearer secret"}).status_code == 401
    assert client.post("/warehouse/snowflake", json={"data": [[0, "standup"]]}, headers={"sf-custom-authorization": "Bearer secret"}).status_code == 400
