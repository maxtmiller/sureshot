"""Build a Router from environment variables, so a deployment can swap models without code changes."""
import json
import os
from hunch.backends import ChatDecider, ClaudeAPI, ClaudeCLI, ChatReasoner, default_system2
from hunch.router import Router
from hunch.store import Store

DEFAULT_URL = "https://api.openai.com/v1"


def _chat(cls, n):
    env = os.environ
    extra = json.loads(env.get(f"HUNCH_SYSTEM{n}_EXTRA", "{}"))
    return cls(env[f"HUNCH_SYSTEM{n}_MODEL"], env.get(f"HUNCH_SYSTEM{n}_URL", DEFAULT_URL), env.get(f"HUNCH_SYSTEM{n}_API_KEY"), extra=extra)


def system1_from_env():
    return _chat(ChatDecider, 1)


def system2_from_env():
    kind = os.environ.get("HUNCH_SYSTEM2", "auto")
    model = os.environ.get("HUNCH_SYSTEM2_MODEL")
    if kind == "auto":
        return default_system2()
    if kind == "claude":
        return ClaudeAPI(**({"model": model} if model else {}))
    if kind == "claude-cli":
        return ClaudeCLI(model)
    if kind == "chat":
        return _chat(ChatReasoner, 2)
    if kind == "none":
        return None
    raise ValueError(f"HUNCH_SYSTEM2 must be auto, claude, claude-cli, chat or none, not {kind!r}")


def router_from_env():
    db = os.environ.get("HUNCH_DB")
    return Router(system1_from_env(), system2_from_env(), float(os.environ.get("HUNCH_THRESHOLD", 0.8)),
                  Store(db) if db else None, float(os.environ.get("HUNCH_AUDIT", 0)))
