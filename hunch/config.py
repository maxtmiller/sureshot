"""Build a Router from environment variables, so a deployment can swap models without code changes."""
import json
import os
from pathlib import Path
from hunch.local import LocalDecider
from hunch.backends import ChatDecider, VLLMDecider, ChatReasoner
from hunch.cache import Cache
from hunch.router import Router
from hunch.store import Store

def _chat(cls, n, default_url=None):
    env = os.environ
    extra = json.loads(env.get(f"HUNCH_SYSTEM{n}_EXTRA", "{}"))
    url, key = env.get(f"HUNCH_SYSTEM{n}_URL", default_url), env.get(f"HUNCH_SYSTEM{n}_API_KEY")
    provider = env.get(f"HUNCH_SYSTEM{n}_PROVIDER")
    return cls(env[f"HUNCH_SYSTEM{n}_MODEL"], url, key, extra=extra, **({"provider": provider} if provider else {}))


def system1_from_env():
    """The LLM System 1, behind a trained LocalDecider tier when HUNCH_LOCAL_MODEL names a saved one."""
    kind = os.environ.get("HUNCH_SYSTEM1", "chat")
    if kind == "chat":
        llm = _chat(ChatDecider, 1)
    elif kind == "vllm":
        llm = _chat(VLLMDecider, 1, "http://localhost:8000/v1")
    else:
        raise ValueError(f"HUNCH_SYSTEM1 must be chat or vllm, not {kind!r}")
    return [LocalDecider.load(os.environ["HUNCH_LOCAL_MODEL"]), llm] if os.environ.get("HUNCH_LOCAL_MODEL") else llm


def default_system2():
    """The first System 2 that is configured: a chat model when HUNCH_SYSTEM2_URL or HUNCH_SYSTEM2_PROVIDER is set, else Claude through the API
    when the SDK is installed and a credential is set, else the local claude CLI if installed, else None."""
    if os.environ.get("HUNCH_SYSTEM2_URL") or os.environ.get("HUNCH_SYSTEM2_PROVIDER"):
        return _chat(ChatReasoner, 2)
    from importlib.util import find_spec

    if find_spec("anthropic") and (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        from hunch.claude import ClaudeAPI

        return ClaudeAPI()
    from hunch.claude import ClaudeCLI

    cli = ClaudeCLI()
    return cli if Path(cli.command).exists() else None


def system2_from_env():
    kind = os.environ.get("HUNCH_SYSTEM2", "auto")
    model = os.environ.get("HUNCH_SYSTEM2_MODEL")
    if kind == "auto":
        return default_system2()
    if kind == "claude":
        from hunch.claude import ClaudeAPI

        return ClaudeAPI(**({"model": model} if model else {}))
    if kind == "claude-cli":
        from hunch.claude import ClaudeCLI

        return ClaudeCLI(model)
    if kind == "chat":
        return _chat(ChatReasoner, 2)
    if kind == "none":
        return None
    raise ValueError(f"HUNCH_SYSTEM2 must be auto, claude, claude-cli, chat or none, not {kind!r}")


def router_from_env():
    db, cache = os.environ.get("HUNCH_DB"), os.environ.get("HUNCH_CACHE")
    ttl = os.environ.get("HUNCH_CACHE_TTL")
    return Router(system1_from_env(), system2_from_env(), float(os.environ.get("HUNCH_THRESHOLD", 0.8)),
                  Store(db) if db else None, float(os.environ.get("HUNCH_AUDIT", 0)),
                  cache=Cache(cache, float(ttl) if ttl else None) if cache else None)
