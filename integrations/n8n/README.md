# n8n-nodes-sureshot

An [n8n](https://n8n.io) community node for [sureshot](https://github.com/maxtmiller/sureshot), which classifies text
with a fast, cheap model and asks a stronger one only about what the fast one is unsure of. Its thresholds are
calibrated against your own labels, so the answers it is confident about are ones you can act on.

The node has two operations:

- **Decide** answers a question set about a text, such as which team a ticket goes to and how urgent it is.
- **Feedback** sends back the right answers when a person corrects a decision, which is what sureshot calibrates on.

It also works as a tool for n8n's AI Agent node.

## 1. Run a sureshot server

The node talks to a sureshot server you run, with your own model keys. The quickest start is the Docker image, with
the example question sets:

```bash
curl -O https://raw.githubusercontent.com/maxtmiller/sureshot/main/examples/questions.yaml
docker run -p 8000:8000 -v sureshot-data:/data -v $PWD/questions.yaml:/questions.yaml \
  -e SURESHOT_QUESTIONS=/questions.yaml -e SURESHOT_API_KEY=pick-a-secret \
  -e SURESHOT_SYSTEM1_PROVIDER=openai -e SURESHOT_SYSTEM1_MODEL=gpt-4o-mini -e OPENAI_API_KEY=... \
  -e ANTHROPIC_API_KEY=... \
  ghcr.io/maxtmiller/sureshot
```

See the [sureshot README](https://github.com/maxtmiller/sureshot#readme) for other models, writing your own questions,
and Postgres.

## 2. Install the node

In n8n, open *Settings > Community Nodes*, choose *Install*, and enter `n8n-nodes-sureshot`. Then add a
*Sureshot API* credential with:

- **URL:** where n8n reaches the server, such as `http://sureshot:8000` on the same Docker network.
- **API Key:** the server's `SURESHOT_API_KEY`.

## 3. Use it

A ticket-routing workflow: *Trigger (new ticket)* > *sureshot: Decide* > *Switch* on the team.

In **Decide**, pick the question set from the dropdown (here `ticket_routing`) and map the text, such as
`{{ $json.body }}`. Under *Options*, *Reference* stores your own ID for the item, and turning off *Escalate* skips the
stronger model. The output is the decision:

```json
{
  "id": "5e4f487ecb1c46b0a0ca262469e18f42",
  "answers": {
    "team": {"answer": "billing", "confidence": 0.99, "source": "sureshot", "tier": 0},
    "priority": {"answer": 3, "confidence": 0.99, "source": "sureshot", "tier": 0}
  },
  "escalated": [],
  "unsure": []
}
```

Later nodes read `{{ $json.answers.team.answer }}`. A score's answer is its level's position, so
`priority` 3 is *Urgent* in `Low, Normal, High, Urgent`. `source` is `system2` when the stronger model answered.

When a person corrects the result, **Feedback** takes the decision's `id` and the right answers as question/answer
pairs. An answer is a choice's key (`technical`), a score level's position (`2`), or `true`/`false` for a yes-no
question.

## Links

- [sureshot](https://github.com/maxtmiller/sureshot): how it decides, calibration, and the rest of its API
- [The n8n, Zapier and Make recipe](https://github.com/maxtmiller/sureshot/blob/main/examples/recipes.md#n8n-zapier-and-make)
- [Issues](https://github.com/maxtmiller/sureshot/issues)
- Building the node: [CONTRIBUTING.md](https://github.com/maxtmiller/sureshot/blob/main/integrations/n8n/CONTRIBUTING.md)

MIT licensed ([LICENSE.md](LICENSE.md)), as n8n asks of community nodes; sureshot itself is Apache-2.0.
