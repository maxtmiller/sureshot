# n8n-nodes-sureshot

An [n8n](https://n8n.io) community node for [sureshot](https://github.com/maxtmiller/sureshot): calibrated decisions
from a fast model, escalated to a stronger one when it is unsure.

- **Decide** answers a question set about a text. The set is a dropdown of the server's sets (`SURESHOT_QUESTIONS`).
  The output is the decision: `answers.<question>.answer`, each answer's source and confidence, and the `id` to label.
- **Feedback** saves a decision's true answers (`true`/`false` for a yes-no question, a number for a score's level,
  otherwise a choice's key), which calibration measures the models against.

Install it in n8n under *Settings > Community Nodes* as `n8n-nodes-sureshot`, then add a *sureshot API* credential
with the server's URL and its `SURESHOT_API_KEY`.

## Development

```bash
npm install
npm test        # builds to dist/ with n8n-node and runs the tests
npm run lint    # n8n's community-node rules
```

To try it in a local n8n, mount `dist/` as a custom extension, e.g.
`docker run -p 5678:5678 -v $PWD/dist:/home/node/.n8n/custom/n8n-nodes-sureshot n8nio/n8n`.
Publish with `npm publish` from this folder; it lints and builds first, and refuses while the lint fails.

This package is MIT licensed ([LICENSE.md](LICENSE.md)), as n8n asks of community nodes; sureshot itself is Apache-2.0.
