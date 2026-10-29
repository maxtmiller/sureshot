"""Claude as System 2, through the Anthropic API or the local `claude` CLI. Optional: `pip install "hunch[claude]"`."""
import json
import shutil
import subprocess
from pathlib import Path
from hunch.backends import PROMPT_VERSION, SYSTEM_PROMPT, Outage, prompt
from hunch.questions import answer_schema


def _image_block(data_uri):
    header, _, data = data_uri.partition(",")
    return {"type": "image", "source": {"type": "base64", "media_type": header.removeprefix("data:").split(";")[0], "data": data}}


class ClaudeAPI:
    """Claude through the Anthropic API, constrained to the question schema. Needs ANTHROPIC_API_KEY (or another
    credential the SDK finds)."""

    def __init__(self, model="claude-haiku-5-5", effort="low", client=None):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError('ClaudeAPI needs the Anthropic SDK: pip install "hunch[claude]"') from e
        self.model, self.effort = model, effort
        self.identity = f"claude-api|{model}|{effort}|v{PROMPT_VERSION}"
        self.client = client or anthropic.Anthropic()

    def answer(self, state, questions, images=()):
        import anthropic

        content = [_image_block(i) for i in images] + [{"type": "text", "text": prompt(state, questions)}]
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": answer_schema(questions)}},
                # If a safety classifier declines, the API reruns the request on a fallback model it picks.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError) as e:
            raise Outage(repr(e)) from e
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined: {response.stop_details.category if response.stop_details else 'no category'}")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("Claude ran out of tokens before answering")
        return json.loads(next(b.text for b in response.content if b.type == "text"))


class ClaudeCLI:
    """Claude through the local `claude` CLI (your Claude Code login), with no tools and the question schema.
    For local use: it can't take images and needs Claude Code installed and signed in."""

    def __init__(self, model=None, command=None, timeout=600):
        self.model, self.timeout = model, timeout
        self.identity = f"claude-cli|{model}|v{PROMPT_VERSION}"
        self.command = command or shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude")

    def answer(self, state, questions, images=()):
        if images:
            raise ValueError("ClaudeCLI can't take images; use ClaudeAPI")
        command = [self.command, "-p", "--output-format", "json", "--tools", "", "--no-session-persistence",
                   "--system-prompt", SYSTEM_PROMPT, "--json-schema", json.dumps(answer_schema(questions))]
        if self.model:
            command += ["--model", self.model]
        result = subprocess.run(command, input=prompt(state, questions), capture_output=True, text=True, timeout=self.timeout)
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            data = {}
        if result.returncode or data.get("is_error") or not isinstance(data.get("structured_output"), dict):
            raise RuntimeError(f"claude CLI failed: {(result.stderr or result.stdout).strip()[:300]}")
        return data["structured_output"]
