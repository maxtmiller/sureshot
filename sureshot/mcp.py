"""An MCP server, so agents (Claude Code, Claude Desktop, any MCP client) can decide with sureshot and help label.
`sureshot mcp` runs it over stdio; `sureshot serve` also answers it at /mcp, behind SURESHOT_API_KEY. Needs
sureshot[mcp] (or sureshot[api]).

Labels given through it are saved with source "mcp", and review hides what the models said on the randomly sampled
items: calibration confirms thresholds with those labels, so they must be a person's own answers."""
import json
from typing import Annotated, Any, Literal
from pydantic import Field
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from sureshot.review import REASONS, review_queue
from sureshot.stats import stats as usage_stats

RANDOM = ("sample", "confident")  # the reasons whose labels must not be anchored on a model's answer

INSTRUCTIONS = """sureshot answers classification questions with a fast model and escalates the ones it is unsure about.
Use decide for an answer. review lists questions a person should label; ask the user each one and pass only their
answers to feedback, never your own guess: labels are treated as the truth the models are measured against."""


def server(router, sets):
    """The MCP server. router is a function returning the Router (so the HTTP API can build it on first use); sets is
    {name: questions} from the questions file, whose names decide accepts and which are readable as resources."""
    # The SDK sets up root logging at this level, and at INFO that logs every model API request's URL.
    app = MCPServer("sureshot", instructions=INSTRUCTIONS, log_level="WARNING")

    def call(fn):
        """fn(), with sureshot's errors (a bad question, an unknown decision, System 1 not ready) as the tool's error."""
        try:
            return fn()
        except (ValueError, RuntimeError) as e:
            raise ToolError(str(e)) from e
        except KeyError as e:
            raise ToolError(e.args[0]) from e

    def log():
        store = call(router).log
        if store is None:
            raise ToolError("set SURESHOT_DB to log decisions")
        return store

    names = Literal[tuple(sets)] if sets else str
    questions_help = "the name of a question set" + (f" ({', '.join(sets)})" if sets else "") + \
                     ", or an object of named questions, each {type: choice|score|noul, instructions, criteria}" + \
                     (". May be left out when there is one set" if len(sets) == 1 else "")

    @app.tool()
    def decide(state: Annotated[str, Field(description="the text to decide about")],
               questions: Annotated[names | dict[str, Any] | None, Field(description=questions_help)] = None,
               escalate: Annotated[bool, Field(description="ask System 2 about what System 1 is unsure of")] = True,
               ref: Annotated[str | None, Field(description="your own id for this item, so labels can find it later")] = None) -> dict[str, Any]:
        """Answer named questions about a text: each answer comes with its confidence and source (a System 1 tier, or
        System 2 when System 1 was unsure). The result's id is what feedback labels."""
        return call(lambda: router().decide(state, questions, (), None, escalate, ref))

    @app.tool()
    def feedback(id: Annotated[str, Field(description="the decision id, from decide or review")],
                 labels: Annotated[dict[str, Any], Field(description="{question key: the true answer}: a choice's key, a score level's 0-based index, or true/false")]) -> dict[str, Any]:
        """Save the true answers a person gave to a decision's questions. Only pass answers the user gave you."""
        store = log()
        for key, answer in labels.items():
            call(lambda: store.label(id, key, answer, source="mcp"))
        return {"ok": True}

    @app.tool()
    def review(limit: Annotated[int, Field(ge=1, le=500)] = 20,
               reason: Annotated[Literal[REASONS] | None, Field(description="only this kind of item")] = None,
               sample: Annotated[float, Field(ge=0, le=1, description="the random sample rate you label at")] = 0.0,
               confident: Annotated[float, Field(ge=0, le=1, description="the random rate of confident answers you label at")] = 0.0) -> list[dict[str, Any]]:
        """The questions a person's label would help most, most useful first: {id, key, reason, state, question} and,
        except for randomly sampled items, what the models answered. Ask the user each one, then call feedback."""
        store = log()
        items = call(lambda: review_queue(store, limit, (reason,) if reason else REASONS, sample=sample, confident=confident))
        hidden = ("answer", "source", "candidates")
        return [{k: v for k, v in item.items() if k not in hidden} if item["reason"] in RANDOM else item for item in items]

    @app.tool()
    def stats(since: Annotated[float, Field(description="unix time")] = 0.0,
              min_samples: Annotated[int, Field(ge=1)] = 30,
              costs: Annotated[dict[str, float] | None, Field(description='cost per question of each stage, e.g. {"tier 0": 0, "system2": 0.004}')] = None,
              sample: Annotated[float, Field(ge=0, le=1, description="the random sample rate your labels came from")] = 0.0,
              confident: Annotated[float, Field(ge=0, le=1, description="the random rate of confident answers your labels came from")] = 0.0) -> dict[str, Any]:
        """Usage, accuracy, cost, latency and drift of the decisions logged since a time."""
        if costs is not None and "system2" not in costs:
            raise ToolError('costs needs a "system2" cost per question')
        store = log()
        return call(lambda: usage_stats(store, since, min_samples, costs=costs, sample=sample, confident=confident))

    for name, questions in sets.items():
        app.resource(f"sureshot://questions/{name}", name=name, description=f"the {name!r} question set", mime_type="application/json")(
            _text(json.dumps(questions, indent=1)))
    return app


def _text(text):
    return lambda: text
