// Runs the app with Zapier's app tester against a stand-in server, or a real sureshot when SURESHOT_URL and
// SURESHOT_API_KEY are set (its questions file must have the ticket_routing set from examples/questions.yaml).
const { test, before, after } = require('node:test');
const assert = require('node:assert');
const http = require('node:http');
const zapier = require('zapier-platform-core');
const { prepareApp, validateApp } = require('zapier-platform-core/src/tools/schema');
const App = require('../index');
const { labelValue } = require('../labels');

const appTester = zapier.createAppTester(App);
const received = [];
let server;
let authData = { url: process.env.SURESHOT_URL, apiKey: process.env.SURESHOT_API_KEY };
const live = Boolean(process.env.SURESHOT_URL);

before(async () => {
  if (live) return;
  server = http.createServer((req, res) => {
    let body = '';
    req.on('data', (chunk) => (body += chunk));
    req.on('end', () => {
      const reply = (status, data) => res.writeHead(status, { 'Content-Type': 'application/json' }).end(JSON.stringify(data));
      if (req.headers.authorization !== 'Bearer secret') return reply(401, { detail: 'missing or wrong bearer token' });
      received.push({ method: req.method, url: req.url, body: body && JSON.parse(body) });
      if (req.url === '/questions') return reply(200, { sets: [{ name: 'ticket_routing', questions: { team: {}, priority: {} } }] });
      if (req.url === '/decide') return reply(200, { id: 'd1', answers: { team: { answer: 'billing', source: 'sureshot', confidence: 0.99 } } });
      if (req.url === '/feedback') return reply(200, { ok: true });
      return reply(404, { detail: 'Not Found' });
    });
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  authData = { url: `http://127.0.0.1:${server.address().port}/`, apiKey: 'secret' };
});

after(() => server && server.close());

test('the app definition passes Zapier\'s schema, as `zapier validate` checks it', () => {
  assert.deepStrictEqual(validateApp(prepareApp(App)), []);
});

test('labels typed in a Zap get the types sureshot wants', () => {
  assert.deepStrictEqual(['billing', 'True', '2', ' 0 '].map(labelValue), ['billing', true, 2, 0]);
});

test('the connection test and the dropdown list the question sets', async () => {
  const sets = await appTester(App.authentication.test, { authData });
  assert.ok(sets.some((set) => set.id === 'ticket_routing' && set.name === 'ticket_routing'));
  await assert.rejects(appTester(App.authentication.test, { authData: { ...authData, apiKey: 'wrong' } }), /sureshot answered 401: missing or wrong bearer token/);
});

test('decide, then label the decision', async () => {
  const text = 'I was charged twice for my subscription this month, please refund one.';
  const decision = await appTester(App.creates.decide.operation.perform, {
    authData, inputData: { question_set: 'ticket_routing', text, ref: 'zap-1', escalate: true },
  });
  assert.strictEqual(decision.answers.team.answer, 'billing');
  assert.ok(decision.id);
  const ok = await appTester(App.creates.feedback.operation.perform, {
    authData, inputData: { decision_id: decision.id, labels: { team: 'billing', priority: '2' } },
  });
  assert.deepStrictEqual(ok, { ok: true });
  if (!live) {
    assert.deepStrictEqual(received.find((r) => r.url === '/decide').body, { state: text, questions: 'ticket_routing', ref: 'zap-1', escalate: true });
    assert.deepStrictEqual(received.find((r) => r.url === '/feedback').body, { id: 'd1', labels: { team: 'billing', priority: 2 } });
  }
});
