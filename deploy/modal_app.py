"""The HTTP API on a Modal CPU container: `modal deploy deploy/modal_app.py`.

Needs a Modal secret named "system1" with PPLX_DECIDER_URL, PPLX_DECIDER_API_KEY, ANTHROPIC_API_KEY and
SYSTEM1_API_KEY. The claude CLI isn't in the container, so System 2 there is always the Anthropic API."""
import modal

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("httpx", "anthropic", "fastapi", "python-dotenv")
    .add_local_python_source("system1")
)
app = modal.App("system1", image=image)


@app.function(secrets=[modal.Secret.from_name("system1")], scaledown_window=300)
@modal.asgi_app()
def web():
    from system1.api import app as api

    return api
