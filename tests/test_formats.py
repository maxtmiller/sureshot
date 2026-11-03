import json
import httpx
import pytest
from sureshot.formats import anthropic_request, gemini_request, openai_request
from sureshot.router import Router
from sureshot.schema import from_schema

SCHEMA = {
    "type": "object",
    "properties": {
        "team": {"type": "string", "enum": ["billing", "support"], "description": "Which team should handle it?"},
        "urgent": {"type": "boolean"},
        "priority": {"type": "integer", "minimum": 1, "maximum": 5},
        "size": {"type": "integer", "enum": [10, 20, 40]},
    },
    "required": ["team", "urgent", "priority", "size"],
    "additionalProperties": False,
}


class FirstOption:
    """System 1 that is sure of the first option of a choice, yes, and the top score level."""

    def decide(self, state, questions, images=()):
        def answer(q):
            return next(iter(q["criteria"])) if q["type"] == "choice" else True if q["type"] == "noul" else len(q["criteria"]) - 1

        return {k: {"answer": answer(q), "confidence": 0.99, "probabilities": {}} for k, q in questions.items()}


def test_schema_fields_map_to_questions():
    questions, encode = from_schema(SCHEMA, "ticket.")
    assert questions["ticket.team"] == {"type": "choice", "instructions": "Which team should handle it?", "criteria": {"billing": "", "support": ""}}
    assert questions["ticket.urgent"] == {"type": "noul"}
    assert questions["ticket.priority"]["criteria"] == ["1", "2", "3", "4", "5"] and questions["ticket.size"]["criteria"] == ["10", "20", "40"]
    answers = {"ticket.team": {"answer": "support"}, "ticket.urgent": {"answer": False}, "ticket.priority": {"answer": 0}, "ticket.size": {"answer": 2}}
    assert encode(answers) == {"team": "support", "urgent": False, "priority": 1, "size": 40}


def test_pydantic_style_refs_resolve():
    schema = {"$defs": {"Team": {"enum": ["billing", "support"], "title": "Team", "type": "string"}},
              "properties": {"team": {"$ref": "#/$defs/Team"}, "other": {"allOf": [{"$ref": "#/$defs/Team"}], "description": "Second team"}},
              "required": ["team", "other"], "title": "Ticket", "type": "object"}
    questions, _ = from_schema(schema)
    assert questions["team"]["criteria"] == {"billing": "", "support": ""} and questions["team"]["instructions"] == "Team"
    assert questions["other"]["instructions"] == "Second team"


@pytest.mark.parametrize("prop", [{"type": "string"}, {"type": "number"}, {"type": "integer", "minimum": 0, "maximum": 100}, {"type": "integer"},
                                  {"anyOf": [{"type": "string", "enum": ["a"]}, {"type": "null"}]}, {"type": "object", "properties": {}},
                                  {"type": "array", "items": {"type": "boolean"}}, {"enum": ["a", 1]}, {"$ref": "https://example.test/x"}])
def test_fields_that_dont_map_leave_the_schema_unanswered(prop):
    assert from_schema({"type": "object", "properties": {"ok": {"type": "boolean"}, "bad": prop}}) is None


def _chat(**more):
    return {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "Route support tickets."}, {"role": "user", "content": "I was charged twice"}]} | more


def test_openai_requests_are_read_from_response_format_or_a_forced_tool():
    fmt = {"type": "json_schema", "json_schema": {"name": "ticket", "schema": SCHEMA, "strict": True}}
    request = openai_request(_chat(response_format=fmt))
    assert request["state"] == "Route support tickets.\n\nI was charged twice" and request["tool"] is None and "ticket.team" in request["questions"]
    tools = [{"type": "function", "function": {"name": "route", "parameters": SCHEMA}}]
    assert openai_request(_chat(tools=tools, tool_choice={"type": "function", "function": {"name": "route"}}))["tool"] == "route"
    assert openai_request(_chat(tools=tools, tool_choice="required"))["questions"].keys() == {f"route.{k}" for k in SCHEMA["properties"]}
    assert openai_request(_chat(tools=tools)) is None
    assert openai_request(_chat(response_format={"type": "json_object"})) is None
    assert openai_request(_chat(response_format=fmt, stream=True))["questions"] == request["questions"]
    image = {"role": "user", "content": [{"type": "text", "text": "Is this urgent?"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}
    assert openai_request(_chat(response_format=fmt) | {"messages": [image]})["images"] == ["data:image/png;base64,AAAA"]
    image["content"][1]["image_url"]["url"] = "https://example.test/a.png"
    assert openai_request(_chat(response_format=fmt) | {"messages": [image]}) is None


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import sureshot.api as api
    from sureshot import Store

    monkeypatch.setenv("SURESHOT_API_KEY", "secret")
    for fmt in ("OPENAI", "ANTHROPIC", "GEMINI"):
        monkeypatch.delenv(f"SURESHOT_{fmt}_PASSTHROUGH_URL", raising=False)
    monkeypatch.setattr(api, "_router", Router(FirstOption(), None, log=Store(tmp_path / "sureshot.db")))
    return TestClient(api.app, headers={"Authorization": "Bearer secret"})


def test_chat_completions_answer_a_schema_and_label_by_header(client):
    fmt = {"type": "json_schema", "json_schema": {"name": "ticket", "schema": SCHEMA}}
    response = client.post("/v1/chat/completions", json=_chat(response_format=fmt))
    body, meta = response.json(), json.loads(response.headers["x-sureshot"])
    assert json.loads(body["choices"][0]["message"]["content"]) == {"team": "billing", "urgent": True, "priority": 5, "size": 40}
    assert body["model"] == "gpt-4o-mini" and body["choices"][0]["finish_reason"] == "stop"
    assert meta["answers"]["ticket.team"] == {"source": "sureshot", "confidence": 0.99}
    assert client.post("/feedback", json={"id": meta["id"], "labels": {"ticket.team": "support"}}).status_code == 200
    tools = [{"type": "function", "function": {"name": "route", "parameters": SCHEMA}}]
    call = client.post("/v1/chat/completions", json=_chat(tools=tools, tool_choice="required")).json()["choices"][0]
    assert call["finish_reason"] == "tool_calls" and json.loads(call["message"]["tool_calls"][0]["function"]["arguments"])["team"] == "billing"
    assert client.post("/v1/chat/completions", json=_chat(response_format=fmt), headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_other_requests_pass_through_or_are_refused(client, monkeypatch):
    import sureshot.api as api

    assert client.post("/v1/chat/completions", json=_chat()).status_code == 422
    seen = {}

    def post(url, json, headers, timeout):
        seen.update(url=url, json=json, headers=headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]}, request=httpx.Request("POST", url))

    monkeypatch.setattr(api, "post", post)
    monkeypatch.setenv("SURESHOT_OPENAI_PASSTHROUGH_URL", "https://upstream.test/v1/")
    monkeypatch.setenv("SURESHOT_OPENAI_PASSTHROUGH_API_KEY", "up")
    assert client.post("/v1/chat/completions", json=_chat()).json()["choices"][0]["message"]["content"] == "hi"
    assert seen == {"url": "https://upstream.test/v1/chat/completions", "json": _chat(), "headers": {"Authorization": "Bearer up"}}


def test_streaming_requests_pass_through_as_a_stream(client, monkeypatch):
    import sureshot.api as api

    seen, sent = [], []

    def post_stream(url, json, headers, timeout):
        seen.append(url)
        sent.append(httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=httpx.ByteStream(b'data: {"a": 1}\n\ndata: [DONE]\n\n')))
        return sent[-1]

    monkeypatch.setattr(api, "post_stream", post_stream)
    assert client.post("/v1/chat/completions", json=_chat(stream=True)).status_code == 422
    for fmt, url in (("OPENAI", "https://openai.test/v1"), ("ANTHROPIC", "https://anthropic.test"), ("GEMINI", "https://gemini.test")):
        monkeypatch.setenv(f"SURESHOT_{fmt}_PASSTHROUGH_URL", url)
    response = client.post("/v1/chat/completions", json=_chat(stream=True))
    assert response.headers["content-type"].startswith("text/event-stream") and response.text == 'data: {"a": 1}\n\ndata: [DONE]\n\n'
    client.post("/v1/messages", json={"model": "claude-haiku-5-5", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}], "stream": True})
    client.post("/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=secret", json={"contents": GEMINI_BODY["contents"]})
    assert seen == ["https://openai.test/v1/chat/completions", "https://anthropic.test/v1/messages",
                    "https://gemini.test/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"] and all(r.is_closed for r in sent)


class Endless(httpx.SyncByteStream):
    def __iter__(self):
        while True:
            yield b"data: {}\n\n"


@pytest.mark.parametrize("spec", ["2.3", "2.4"])
def test_a_relayed_stream_closes_upstream_when_the_client_hangs_up(spec):
    """Below ASGI 2.4 Starlette hears the disconnect and stops; from 2.4 a send fails. Either way the upstream closes."""
    import asyncio
    from sureshot.api import Relay

    upstream, sends = httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Endless()), []

    async def receive():
        await asyncio.sleep(0.05)
        return {"type": "http.disconnect"}

    async def send(message):
        sends.append(message)
        if spec == "2.4" and len(sends) > 3:
            raise OSError("client gone")
        await asyncio.sleep(0.001)

    try:
        asyncio.run(Relay(upstream)({"type": "http", "asgi": {"spec_version": spec}}, receive, send))
    except Exception as e:
        assert spec == "2.4" and type(e).__name__ == "ClientDisconnect"
    assert upstream.is_closed and len(sends) > 1


def test_the_openai_sdk_parses_a_pydantic_model(client):
    openai = pytest.importorskip("openai")
    from typing import Literal
    from pydantic import BaseModel, Field

    class Ticket(BaseModel):
        team: Literal["billing", "support"] = Field(description="Which team should handle it?")
        urgent: bool

    sdk = openai.OpenAI(base_url="http://testserver/v1", api_key="secret", http_client=client)
    completion = sdk.beta.chat.completions.parse(model="gpt-4o-mini", messages=_chat()["messages"], response_format=Ticket)
    assert completion.choices[0].message.parsed == Ticket(team="billing", urgent=True)


def test_anthropic_requests_are_read_from_output_config_or_a_forced_tool():
    body = {"model": "claude-haiku-5-5", "max_tokens": 100, "system": [{"type": "text", "text": "Route support tickets."}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "I was charged twice"},
                                                      {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]}
    request = anthropic_request(body | {"output_config": {"format": {"type": "json_schema", "schema": SCHEMA | {"title": "Ticket"}}}})
    assert request["state"] == "Route support tickets.\n\nI was charged twice" and request["images"] == ["data:image/png;base64,AAAA"]
    assert "Ticket.team" in request["questions"] and request["tool"] is None
    tools = [{"name": "route", "input_schema": SCHEMA}]
    assert anthropic_request(body | {"tools": tools, "tool_choice": {"type": "tool", "name": "route"}})["tool"] == "route"
    assert anthropic_request(body | {"tools": tools, "tool_choice": {"type": "any"}})["tool"] == "route"
    assert anthropic_request(body | {"tools": tools}) is None
    assert anthropic_request(body | {"tools": tools, "tool_choice": {"type": "any"}, "stream": True})["tool"] == "route"


def test_anthropic_sdk_constraints_moved_into_descriptions_still_map():
    prop = {"type": "integer", "description": "Priority\n\n{maximum: 5, minimum: 1}", "title": "P"}
    questions, encode = from_schema({"type": "object", "properties": {"p": prop}})
    assert questions["p"]["criteria"] == ["1", "2", "3", "4", "5"] and encode({"p": {"answer": 4}}) == {"p": 5}


# What google-genai 1.30 sends for generate_content(config={"response_schema": Ticket, "system_instruction": ...}).
GEMINI_BODY = {
    "contents": [{"parts": [{"text": "I was charged twice"}], "role": "user"}],
    "systemInstruction": {"parts": [{"text": "Route tickets."}], "role": "user"},
    "generationConfig": {"responseMimeType": "application/json", "responseSchema": {
        "properties": {"team": {"enum": ["billing", "support"], "title": "Team", "type": "STRING"}, "urgent": {"title": "Urgent", "type": "BOOLEAN"},
                       "priority": {"maximum": 5.0, "minimum": 1.0, "title": "Priority", "type": "INTEGER"}},
        "propertyOrdering": ["team", "urgent", "priority"], "required": ["team", "urgent", "priority"], "title": "Ticket", "type": "OBJECT"}},
}


def test_gemini_requests_are_read_from_either_schema_field():
    request = gemini_request(GEMINI_BODY)
    assert request["state"] == "Route tickets.\n\nI was charged twice" and request["questions"]["Ticket.priority"]["criteria"] == ["1", "2", "3", "4", "5"]
    snake = {"contents": [{"parts": [{"text": "hi"}, {"inline_data": {"mime_type": "image/png", "data": "AAAA"}}]}],
             "generation_config": {"response_json_schema": SCHEMA}}
    assert gemini_request(snake)["images"] == ["data:image/png;base64,AAAA"] and "response.team" in gemini_request(snake)["questions"]
    assert gemini_request({"contents": GEMINI_BODY["contents"]}) is None


def test_gemini_requests_are_read_from_a_forced_function():
    # What google-genai 2.34 sends for a FunctionDeclaration with parameters, and mode ANY naming it.
    route = {"name": "route", "parameters": GEMINI_BODY["generationConfig"]["responseSchema"]}
    other = {"name": "other", "parametersJsonSchema": SCHEMA}
    body = {"contents": GEMINI_BODY["contents"], "tools": [{"functionDeclarations": [route, other]}, {"googleSearch": {}}],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["route"]}}}
    request = gemini_request(body)
    assert request["tool"] == "route" and request["questions"].keys() == {"route.team", "route.urgent", "route.priority"}
    snake = {"contents": GEMINI_BODY["contents"], "tools": [{"function_declarations": [other]}],
             "tool_config": {"function_calling_config": {"mode": "ANY"}}}
    assert gemini_request(snake)["tool"] == "other" and "other.team" in gemini_request(snake)["questions"]
    assert gemini_request(body | {"toolConfig": {"functionCallingConfig": {"mode": "ANY"}}}) is None
    assert gemini_request(body | {"toolConfig": {"functionCallingConfig": {"mode": "AUTO", "allowedFunctionNames": ["route"]}}}) is None
    assert gemini_request({k: v for k, v in body.items() if k != "toolConfig"}) is None


def test_anthropic_and_gemini_routes(client, monkeypatch):
    import sureshot.api as api

    body = {"model": "claude-haiku-5-5", "max_tokens": 100, "messages": [{"role": "user", "content": "I was charged twice"}]}
    tools = {"tools": [{"name": "route", "input_schema": SCHEMA}], "tool_choice": {"type": "any"}}
    message = client.post("/v1/messages", json=body | tools, headers={"Authorization": "", "x-api-key": "secret"}).json()
    assert message["stop_reason"] == "tool_use" and message["content"][0]["input"] == {"team": "billing", "urgent": True, "priority": 5, "size": 40}
    assert client.post("/v1/messages", json=body | tools, headers={"Authorization": "", "x-api-key": "wrong"}).status_code == 401
    response = client.post("/v1beta/models/gemini-2.5-flash:generateContent", json=GEMINI_BODY, headers={"Authorization": "", "x-goog-api-key": "secret"})
    assert json.loads(response.json()["candidates"][0]["content"]["parts"][0]["text"]) == {"team": "billing", "urgent": True, "priority": 5}
    assert "Ticket.team" in json.loads(response.headers["x-sureshot"])["answers"]
    assert client.post("/v1beta/models/gemini-2.5-flash:generateContent?key=secret", json=GEMINI_BODY, headers={"Authorization": ""}).status_code == 200
    forced = {"contents": GEMINI_BODY["contents"], "tools": [{"functionDeclarations": [{"name": "route", "parametersJsonSchema": SCHEMA}]}],
              "toolConfig": {"functionCallingConfig": {"mode": "ANY"}}}
    part = client.post("/v1beta/models/gemini-2.5-flash:generateContent?key=secret", json=forced).json()["candidates"][0]["content"]["parts"][0]
    assert part == {"functionCall": {"name": "route", "args": {"team": "billing", "urgent": True, "priority": 5, "size": 40}}}
    seen = {}

    def post(url, json, headers, timeout):
        seen.update(url=url, headers=headers)
        return httpx.Response(200, json={}, request=httpx.Request("POST", url))

    monkeypatch.setattr(api, "post", post)
    monkeypatch.setenv("SURESHOT_ANTHROPIC_PASSTHROUGH_URL", "https://api.anthropic.test")
    monkeypatch.setenv("SURESHOT_ANTHROPIC_PASSTHROUGH_API_KEY", "up")
    client.post("/v1/messages", json=body, headers={"Authorization": "", "x-api-key": "secret", "anthropic-beta": "b1"})
    assert seen == {"url": "https://api.anthropic.test/v1/messages", "headers": {"anthropic-version": "2023-06-01", "anthropic-beta": "b1", "x-api-key": "up"}}
    monkeypatch.setenv("SURESHOT_GEMINI_PASSTHROUGH_URL", "https://gemini.test")
    client.post("/v1/models/gemini-2.5-flash:generateContent", json={"contents": GEMINI_BODY["contents"]}, headers={"Authorization": "", "x-goog-api-key": "secret"})
    assert seen == {"url": "https://gemini.test/v1/models/gemini-2.5-flash:generateContent", "headers": {}}


def test_the_anthropic_sdk_parses_a_pydantic_model(client):
    anthropic = pytest.importorskip("anthropic")
    from enum import Enum
    from pydantic import BaseModel, Field

    class Team(str, Enum):
        billing = "billing"
        support = "support"

    class Ticket(BaseModel):
        team: Team
        urgent: bool
        priority: int = Field(ge=1, le=5)

    sdk = anthropic.Anthropic(base_url="http://testserver", api_key="secret", http_client=client)
    message = sdk.messages.parse(model="claude-haiku-5-5", max_tokens=100, messages=[{"role": "user", "content": "I was charged twice"}], output_format=Ticket)
    assert message.parsed_output == Ticket(team="billing", urgent=True, priority=5)


def _events(response):
    """(event name, data) of each server-sent event."""
    events = []
    for block in response.text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((fields.get("event"), fields["data"] if fields["data"] == "[DONE]" else json.loads(fields["data"])))
    return events


def test_streaming_requests_get_the_answer_as_one_chunk(client):
    fmt = {"type": "json_schema", "json_schema": {"name": "ticket", "schema": SCHEMA}}
    response = client.post("/v1/chat/completions", json=_chat(response_format=fmt, stream=True, stream_options={"include_usage": True}))
    assert response.headers["content-type"].startswith("text/event-stream") and "ticket.team" in json.loads(response.headers["x-sureshot"])["answers"]
    chunks = [data for _, data in _events(response)]
    assert json.loads(chunks[0]["choices"][0]["delta"]["content"])["team"] == "billing" and chunks[1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[2]["choices"] == [] and chunks[2]["usage"]["total_tokens"] == 0 and chunks[3] == "[DONE]"
    tools = [{"type": "function", "function": {"name": "route", "parameters": SCHEMA}}]
    chunks = [data for _, data in _events(client.post("/v1/chat/completions", json=_chat(tools=tools, tool_choice="required", stream=True)))]
    assert chunks[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0 and "usage" not in chunks[0] and len(chunks) == 3
    body = {"model": "claude-haiku-5-5", "max_tokens": 100, "messages": [{"role": "user", "content": "I was charged twice"}], "stream": True,
            "tools": [{"name": "route", "input_schema": SCHEMA}], "tool_choice": {"type": "any"}}
    events = _events(client.post("/v1/messages", json=body))
    assert [name for name, _ in events] == ["message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
    assert json.loads(events[2][1]["delta"]["partial_json"])["team"] == "billing" and events[4][1]["delta"]["stop_reason"] == "tool_use"
    url = "/v1beta/models/gemini-2.5-flash:streamGenerateContent"
    events = _events(client.post(url + "?alt=sse&key=secret", json=GEMINI_BODY))
    assert json.loads(events[0][1]["candidates"][0]["content"]["parts"][0]["text"])["team"] == "billing"
    assert json.loads(client.post(url + "?key=secret", json=GEMINI_BODY).json()[0]["candidates"][0]["content"]["parts"][0]["text"])["urgent"] is True
    assert client.post(url + "?alt=sse&key=secret", json={"contents": GEMINI_BODY["contents"]}).status_code == 422


def test_the_sdks_parse_a_streamed_answer(client):
    openai, anthropic = pytest.importorskip("openai"), pytest.importorskip("anthropic")
    from typing import Literal
    from pydantic import BaseModel

    class Ticket(BaseModel):
        team: Literal["billing", "support"]
        urgent: bool

    sdk = openai.OpenAI(base_url="http://testserver/v1", api_key="secret", http_client=client)
    with sdk.chat.completions.stream(model="gpt-4o-mini", messages=_chat()["messages"], response_format=Ticket) as stream:
        assert stream.get_final_completion().choices[0].message.parsed == Ticket(team="billing", urgent=True)
    tool = {"type": "function", "function": {"name": "route", "parameters": SCHEMA}}
    chunks = list(sdk.chat.completions.create(model="gpt-4o-mini", messages=_chat()["messages"], tools=[tool], tool_choice="required", stream=True))
    assert json.loads(chunks[0].choices[0].delta.tool_calls[0].function.arguments)["team"] == "billing"
    sdk = anthropic.Anthropic(base_url="http://testserver", api_key="secret", http_client=client)
    with sdk.messages.stream(model="claude-haiku-5-5", max_tokens=100, messages=[{"role": "user", "content": "I was charged twice"}],
                             tools=[{"name": "route", "input_schema": SCHEMA}], tool_choice={"type": "any"}) as stream:
        assert stream.get_final_message().content[0].input == {"team": "billing", "urgent": True, "priority": 5, "size": 40}
