# Developing n8n-nodes-sureshot

```bash
npm install
npm test        # builds to dist/ with n8n-node and runs the tests
npm run lint    # n8n's community-node rules
```

To try it in a local n8n, mount `dist/` as a custom extension, e.g.
`docker run -p 5678:5678 -v $PWD/dist:/home/node/.n8n/custom/n8n-nodes-sureshot n8nio/n8n`.

To publish, bump `version` in package.json, then run `npm publish --access public --auth-type=web` from this folder.
It lints and builds first, and refuses while the lint fails. npm stages the version and validates it for a minute or
two before it goes live.
