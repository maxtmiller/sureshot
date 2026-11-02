import json
import anyio
from mcp import Client
from sureshot import Store
from sureshot.mcp import server
from sureshot.router import Router
from test_router import QUESTIONS, FakeClaude, FakeDecider


def _router(tmp_path):
    return Router(FakeDecider(), FakeClaude(), log=Store(tmp_path / "sureshot.db"), questions={"events": QUESTIONS})


def _session(router, steps):
    async def run():
        async with Client(server(lambda: router, router.questions)) as client:
            return await steps(client)
    return anyio.run(run)


def test_decide_by_set_name_and_read_the_sets(tmp_path):
    async def steps(client):
        tools = {t.name: t for t in (await client.list_tools()).tools}
        decided = await client.call_tool("decide", {"state": "standup", "questions": "events", "ref": "cal-1"})
        unknown = await client.call_tool("decide", {"state": "standup", "questions": "nope"})
        resources = [str(r.uri) for r in (await client.list_resources()).resources]
        text = (await client.read_resource("sureshot://questions/events")).contents[0].text
        return tools, decided, unknown, resources, text

    tools, decided, unknown, resources, text = _session(_router(tmp_path), steps)
    assert set(tools) == {"decide", "feedback", "review", "stats"}
    assert {"const": "events", "type": "string"} in tools["decide"].input_schema["properties"]["questions"]["anyOf"]
    assert not decided.is_error and decided.structured_content["escalated"] == ["importance"]
    assert unknown.is_error
    assert resources == ["sureshot://questions/events"] and json.loads(text) == QUESTIONS


def test_review_hides_model_answers_on_random_items_and_feedback_labels_as_mcp(tmp_path):
    router = _router(tmp_path)
    decision = router.decide("standup", "events")["id"]

    async def steps(client):
        sampled = (await client.call_tool("review", {"sample": 1})).structured_content["result"]
        disagreed = (await client.call_tool("review", {"reason": "disagreement"})).structured_content["result"]
        bad = await client.call_tool("feedback", {"id": decision, "labels": {"needs_prep": "maybe"}})
        missing = await client.call_tool("feedback", {"id": "nope", "labels": {"needs_prep": True}})
        ok = await client.call_tool("feedback", {"id": decision, "labels": {"needs_prep": False}})
        return sampled, disagreed, bad, missing, ok

    sampled, disagreed, bad, missing, ok = _session(router, steps)
    assert len(sampled) == 3 and all(item["reason"] == "sample" and not {"answer", "source", "candidates"} & set(item) for item in sampled)
    assert [item["key"] for item in disagreed] == ["importance"] and [c["who"] for c in disagreed[0]["candidates"]] == ["tier 0", "system2"]
    assert bad.is_error and "not a valid answer" in bad.content[0].text
    assert missing.is_error and "no decision nope" in missing.content[0].text
    assert not ok.is_error and router.log.labels() == {(decision, "needs_prep"): False}
    with router.log._db() as db:
        assert db.execute("SELECT source FROM labels").fetchall() == [("mcp",)]


def test_stats_and_a_router_without_a_log(tmp_path):
    router = _router(tmp_path)
    router.decide("standup", "events")

    async def steps(client):
        return await client.call_tool("stats", {}), await client.call_tool("stats", {"costs": {"tier 0": 0}})

    report, bad = _session(router, steps)
    assert not report.is_error and report.structured_content["decisions"] == 1
    assert bad.is_error and 'needs a "system2" cost' in bad.content[0].text

    async def unlogged(client):
        return await client.call_tool("review", {})

    result = _session(Router(FakeDecider(), FakeClaude(), questions={"events": QUESTIONS}), unlogged)
    assert result.is_error and "set SURESHOT_DB" in result.content[0].text


def test_the_api_answers_mcp_with_its_token(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import sureshot.api as api

    monkeypatch.setenv("SURESHOT_API_KEY", "secret")
    monkeypatch.setattr(api, "_router", _router(tmp_path))
    call = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "decide", "arguments": {"state": "standup", "questions": QUESTIONS}}}
    headers = {"Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-06-18"}
    for _ in range(2):  # the app can start again, as each test client starts it
        with TestClient(api.app) as client:
            assert client.post("/mcp", json=call, headers=headers).status_code == 401
            reply = client.post("/mcp", json=call, headers=headers | {"Authorization": "Bearer secret"})
            assert reply.status_code == 200 and reply.json()["result"]["structuredContent"]["escalated"] == ["importance"]
