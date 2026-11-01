"""Other APIs' request formats, answered by the router: an app that asks a model for an object matching a JSON schema
can point its base URL at sureshot instead. Each format has a function that reads a request into {"state", "images",
"questions", "encode", "tool"}, or None when sureshot can't answer it, and one that writes the response to it.

A request is answered only when it asks for a schema whose fields all map to questions (see schema.from_schema), does
not stream, and sends only text and inline images. Its state is the text of every message in order, system prompt
included. Question keys are the schema's (or forced tool's) name and the field, e.g. "ticket.team"."""
import json
import time
import uuid
from sureshot.schema import from_schema


def _mapped(schema, name, texts, images, tool=None):
    mapped = isinstance(schema, dict) and from_schema(schema, f"{name or schema.get('title') or 'response'}.")
    if not mapped:
        return None
    return {"state": "\n\n".join(t for t in texts if t), "images": images, "questions": mapped[0], "encode": mapped[1], "tool": tool}


def _read(contents, part):
    """(texts, images) of every content part, each read by part(...) into ("text", ...) or ("image", data URI), or None
    if any part is something else (an image by URL, a document, a tool call...)."""
    texts, images = [], []
    for content in contents:
        for p in [content] if isinstance(content, str) else content or []:
            got = ("text", p) if isinstance(p, str) else part(p)
            if got is None:
                return None
            (texts if got[0] == "text" else images).append(got[1])
    return texts, images


def summary(result):
    """What the x-sureshot response header carries: the decision id, for POST /feedback, and who answered each question."""
    return {"id": result.get("id"), "answers": {key: {"source": a["source"], "confidence": a.get("confidence")} for key, a in result["answers"].items()}}


def _openai_part(p):
    if p.get("type") == "text":
        return "text", p["text"]
    url = (p.get("image_url") or {}).get("url", "") if p.get("type") == "image_url" else ""
    return ("image", url) if str(url).startswith("data:") else None


def openai_request(body):
    """A chat completion that asks for a schema through response_format, or through one forced function: named in
    tool_choice, or the only tool with tool_choice "required". Several choices (n above 1) aren't answered."""
    if body.get("stream") or (body.get("n") or 1) != 1:
        return None
    fmt, tool = body.get("response_format") or {}, None
    if fmt.get("type") == "json_schema":
        spec = fmt.get("json_schema") or {}
        schema = spec.get("schema")
    else:
        tools = [t["function"] for t in body.get("tools") or [] if t.get("type") == "function" and "function" in t]
        choice = body.get("tool_choice")
        if isinstance(choice, dict) and choice.get("type") == "function":
            tool = next((t for t in tools if t.get("name") == (choice.get("function") or {}).get("name")), None)
        elif choice == "required" and len(tools) == 1:
            tool = tools[0]
        spec, schema = tool or {}, tool and tool.get("parameters")
    read = _read([m.get("content") for m in body.get("messages") or []], _openai_part)
    return read and _mapped(schema, spec.get("name"), *read, tool and tool["name"])


def openai_response(body, request, result):
    """Token usage is reported as zero, since the models sureshot asked are not the one the app named."""
    content = json.dumps(request["encode"](result["answers"]))
    if request["tool"]:
        message = {"role": "assistant", "content": None, "refusal": None,
                   "tool_calls": [{"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function", "function": {"name": request["tool"], "arguments": content}}]}
    else:
        message = {"role": "assistant", "content": content, "refusal": None}
    return {"id": f"chatcmpl-{result.get('id') or uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()), "model": body.get("model", "sureshot"),
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if request["tool"] else "stop", "logprobs": None}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


def _anthropic_part(p):
    if p.get("type") == "text":
        return "text", p["text"]
    source = p.get("source") or {}
    if p.get("type") == "image" and source.get("type") == "base64":
        return "image", f"data:{source['media_type']};base64,{source['data']}"
    return None


def anthropic_request(body):
    """A Messages request that asks for a schema through output_config.format (or the older output_format), or through
    one forced tool: named in tool_choice, or the only tool with tool_choice "any". The schema's title names it."""
    if body.get("stream"):
        return None
    fmt, tool = (body.get("output_config") or {}).get("format") or body.get("output_format") or {}, None
    if fmt.get("type") == "json_schema":
        schema = fmt.get("schema")
    else:
        tools = [t for t in body.get("tools") or [] if "input_schema" in t]
        choice = body.get("tool_choice") or {}
        if choice.get("type") == "tool":
            tool = next((t for t in tools if t.get("name") == choice.get("name")), None)
        elif choice.get("type") == "any" and len(tools) == 1:
            tool = tools[0]
        schema = tool and tool["input_schema"]
    read = _read([body.get("system")] + [m.get("content") for m in body.get("messages") or []], _anthropic_part)
    return read and _mapped(schema, tool and tool["name"], *read, tool and tool["name"])


def anthropic_response(body, request, result):
    answer = request["encode"](result["answers"])
    if request["tool"]:
        content = [{"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:24]}", "name": request["tool"], "input": answer}]
    else:
        content = [{"type": "text", "text": json.dumps(answer)}]
    return {"id": f"msg_{result.get('id') or uuid.uuid4().hex}", "type": "message", "role": "assistant", "model": body.get("model", "sureshot"), "content": content,
            "stop_reason": "tool_use" if request["tool"] else "end_turn", "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}


def _either(d, camel, snake):
    """Gemini's REST API takes camelCase or snake_case field names."""
    return d.get(camel, d.get(snake)) if isinstance(d, dict) else None


def _gemini_part(p):
    if "text" in p:
        return "text", p["text"]
    inline = _either(p, "inlineData", "inline_data")
    return ("image", f"data:{_either(inline, 'mimeType', 'mime_type')};base64,{inline['data']}") if inline else None


def gemini_request(body):
    """A generateContent request that asks for a schema through generationConfig.responseJsonSchema, or
    responseSchema (OpenAPI types such as "STRING" are read as JSON schema ones). Function calling isn't answered."""
    config = _either(body, "generationConfig", "generation_config") or {}
    schema = _either(config, "responseJsonSchema", "response_json_schema") or _either(config, "responseSchema", "response_schema")
    system = _either(body, "systemInstruction", "system_instruction")
    read = _read([(system or {}).get("parts")] + [c.get("parts") for c in body.get("contents") or []], _gemini_part)
    return read and _mapped(schema, None, *read)


def gemini_response(model, request, result):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": json.dumps(request["encode"](result["answers"]))}]}, "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0}, "modelVersion": model, "responseId": result.get("id") or uuid.uuid4().hex}
