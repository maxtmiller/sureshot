"""One System 1 and one System 2 class per major vendor that speaks the chat-completions format, each with the vendor's
endpoint and API-key variable built in: `OpenAIDecider("gpt-4o-mini")`, `GroqReasoner("llama-3.3-70b-versatile")`.

They are the generic `ChatDecider` and `ChatReasoner` with those defaults (and `base_url=` still overrides the
endpoint), so what a vendor supports is up to the vendor: a System 1 needs a model that returns logprobs, and a System 2
one that follows a JSON schema. Run `.check()` on a decider before relying on it. There is no `GroqDecider` because Groq
is reported not to return logprobs. Gemini and Claude have their own protocols and live in `gemini.py` and
`anthropic.py`."""
from sureshot.adapters.chat import ChatDecider, ChatReasoner


def _decider(name, provider, vendor):
    def __init__(self, model, api_key=None, base_url=None, timeout=60, extra=None):
        ChatDecider.__init__(self, model, base_url, api_key, timeout, extra, provider)

    return type(name, (ChatDecider,), {"__init__": __init__, "__module__": __name__,
                                       "__doc__": f"{vendor} as System 1 (a `ChatDecider` for {vendor}'s endpoint; reads the key from its usual environment variable). Needs a model that returns logprobs."})


def _reasoner(name, provider, vendor, **fixed):
    def __init__(self, model, api_key=None, base_url=None, timeout=600, extra=None):
        ChatReasoner.__init__(self, model, base_url, api_key, timeout, extra, provider, **fixed)

    return type(name, (ChatReasoner,), {"__init__": __init__, "__module__": __name__,
                                        "__doc__": f"{vendor} as System 2 (a `ChatReasoner` for {vendor}'s endpoint; reads the key from its usual environment variable)."})


OpenAIDecider = _decider("OpenAIDecider", "openai", "OpenAI")
OpenAIReasoner = _reasoner("OpenAIReasoner", "openai", "OpenAI")
MistralDecider = _decider("MistralDecider", "mistral", "Mistral")
MistralReasoner = _reasoner("MistralReasoner", "mistral", "Mistral")
DeepSeekDecider = _decider("DeepSeekDecider", "deepseek", "DeepSeek")
DeepSeekReasoner = _reasoner("DeepSeekReasoner", "deepseek", "DeepSeek (JSON mode, which is what it offers instead of schemas)", json_mode="object")
XAIDecider = _decider("XAIDecider", "xai", "xAI")
XAIReasoner = _reasoner("XAIReasoner", "xai", "xAI")
GroqReasoner = _reasoner("GroqReasoner", "groq", "Groq")
TogetherDecider = _decider("TogetherDecider", "together", "Together AI")
TogetherReasoner = _reasoner("TogetherReasoner", "together", "Together AI")
FireworksDecider = _decider("FireworksDecider", "fireworks", "Fireworks AI")
FireworksReasoner = _reasoner("FireworksReasoner", "fireworks", "Fireworks AI")
OpenRouterDecider = _decider("OpenRouterDecider", "openrouter", "OpenRouter")
OpenRouterReasoner = _reasoner("OpenRouterReasoner", "openrouter", "OpenRouter")
OllamaDecider = _decider("OllamaDecider", "ollama", "A local Ollama server")
OllamaReasoner = _reasoner("OllamaReasoner", "ollama", "A local Ollama server")

VENDOR_CLASSES = [OpenAIDecider, OpenAIReasoner, MistralDecider, MistralReasoner, DeepSeekDecider, DeepSeekReasoner, XAIDecider, XAIReasoner,
                  GroqReasoner, TogetherDecider, TogetherReasoner, FireworksDecider, FireworksReasoner, OpenRouterDecider, OpenRouterReasoner,
                  OllamaDecider, OllamaReasoner]
