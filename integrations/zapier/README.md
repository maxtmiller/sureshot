# sureshot for Zapier

A [Zapier](https://zapier.com) app for [sureshot](https://github.com/maxtmiller/sureshot): calibrated decisions from a
fast model, escalated to a stronger one when it is unsure.

- **Decide** answers a question set about a text, chosen from a dropdown of the server's sets (`SURESHOT_QUESTIONS`).
  Later steps can use `answers.<question>.answer` and the decision `id`.
- **Give Feedback** saves a decision's true answers as question/answer pairs (`true`/`false` for a yes-no question, a
  number for a score's level, otherwise a choice's key).

The connection asks for the server's URL, which Zapier must reach over HTTPS, and its `SURESHOT_API_KEY`.

## Development

```bash
npm install
npm test                                                      # against a stand-in server
SURESHOT_URL=http://localhost:8000 SURESHOT_API_KEY=... npm test   # against a running sureshot with examples/questions.yaml
```

To use it, install the [Zapier CLI](https://docs.zapier.com/platform/build-cli/overview) (`npm install -g zapier-platform-cli`),
then `zapier login`, `zapier register "sureshot"` once, and `zapier push` from this folder. It is then private to your
account; invite others with `zapier users:add`, or submit it for the public directory.
