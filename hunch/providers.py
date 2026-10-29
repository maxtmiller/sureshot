"""Where common OpenAI-compatible services live, so `provider="groq"` stands in for a URL and a key lookup.

This only saves typing: any endpoint that speaks the chat-completions format works with an explicit base_url. Whether
a service returns the logprobs System 1 needs, or supports the structured outputs System 2 needs, varies by service and
model, so `ChatDecider.check()` before relying on one."""
import os

PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
    "vllm": ("http://localhost:8000/v1", None),
}


def endpoint(base_url=None, api_key=None, provider=None):
    """(base_url, api_key) from what was given: a provider fills in a missing URL and a missing key (read from the
    provider's usual environment variable). Raises ValueError if there is still no URL."""
    if provider is not None:
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}; use one of {sorted(PROVIDERS)} or pass base_url")
        url, key_env = PROVIDERS[provider]
        base_url = base_url or url
        api_key = api_key or (os.environ.get(key_env) if key_env else None)
    if base_url is None:
        raise ValueError(f"give base_url, or provider= one of {sorted(PROVIDERS)}")
    return base_url, api_key
