// sureshot's Zapier app: Decide and Feedback actions, with the server's question sets as a dropdown.
const { version } = require('./package.json');
const { version: platformVersion } = require('zapier-platform-core');

const { labelValue } = require('./labels');

const base = (bundle) => bundle.authData.url.replace(/\/+$/, '');

const authorize = (request, z, bundle) => {
  if (bundle.authData.apiKey) request.headers.Authorization = `Bearer ${bundle.authData.apiKey}`;
  return request;
};

const explain = (response, z) => {
  if (response.status >= 400) {
    const detail = (response.data && response.data.detail) || response.content;
    throw new z.errors.Error(`sureshot answered ${response.status}: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`, 'SureshotError', response.status);
  }
  return response;
};

const questionSets = async (z, bundle) => {
  const response = await z.request({ url: `${base(bundle)}/questions` });
  return response.data.sets.map((set) => ({ id: set.name, name: set.name, questions: Object.keys(set.questions).join(', ') }));
};

const decide = async (z, bundle) => {
  const { question_set, text, ref, escalate } = bundle.inputData;
  const response = await z.request({
    method: 'POST',
    url: `${base(bundle)}/decide`,
    body: { state: text, questions: question_set, ref: ref || null, escalate: escalate !== false && escalate !== 'false' },
  });
  return response.data;
};

const feedback = async (z, bundle) => {
  const labels = Object.fromEntries(Object.entries(bundle.inputData.labels || {}).map(([question, answer]) => [question, labelValue(answer)]));
  const response = await z.request({ method: 'POST', url: `${base(bundle)}/feedback`, body: { id: bundle.inputData.decision_id, labels } });
  return response.data;
};

module.exports = {
  version,
  platformVersion,
  authentication: {
    type: 'custom',
    fields: [
      { key: 'url', label: 'URL', required: true, type: 'string', helpText: 'Where `sureshot serve` runs, such as https://sureshot.example.com' },
      { key: 'apiKey', label: 'API Key', required: true, type: 'password', helpText: "The server's SURESHOT_API_KEY" },
    ],
    test: questionSets,
    connectionLabel: '{{bundle.authData.url}}',
  },
  beforeRequest: [authorize],
  afterResponse: [explain],
  triggers: {
    question_set: {
      key: 'question_set',
      noun: 'Question Set',
      display: { label: 'List Question Sets', description: "Lists the server's question sets, for the Decide dropdown.", hidden: true },
      operation: { perform: questionSets, sample: { id: 'ticket_routing', name: 'ticket_routing', questions: 'team, priority' } },
    },
  },
  creates: {
    decide: {
      key: 'decide',
      noun: 'Decision',
      display: { label: 'Decide', description: 'Answers a question set about a text, escalating to the stronger model when the fast one is unsure.' },
      operation: {
        inputFields: [
          { key: 'question_set', label: 'Question Set', required: true, dynamic: 'question_set.id.name' },
          { key: 'text', label: 'Text', required: true, type: 'text', helpText: 'What to decide about: an email, a ticket, a message.' },
          { key: 'ref', label: 'Reference', required: false, helpText: 'Your own ID for the item (a ticket or message ID), so a later label can find the decision.' },
          { key: 'escalate', label: 'Escalate', required: false, type: 'boolean', default: 'true', helpText: 'Ask the stronger model about what the fast one is unsure of.' },
        ],
        perform: decide,
        sample: {
          id: '274de1049866414ead2279727a7d615f',
          answers: { team: { answer: 'billing', confidence: 0.99, source: 'sureshot', tier: 0 }, priority: { answer: 2, confidence: 0.91, source: 'system2' } },
          unsure: [],
          escalated: ['priority'],
          degraded: false,
        },
        outputFields: [{ key: 'id', label: 'Decision ID' }],
      },
    },
    feedback: {
      key: 'feedback',
      noun: 'Label',
      display: { label: 'Give Feedback', description: "Saves a decision's true answers, which calibration measures the models against." },
      operation: {
        inputFields: [
          { key: 'decision_id', label: 'Decision ID', required: true, helpText: 'The ID a Decide step returned.' },
          {
            key: 'labels', label: 'Labels', required: true, dict: true,
            helpText: "Each question's key (such as team) and its true answer: a choice's key, a score level's 0-based position, or true/false.",
          },
        ],
        perform: feedback,
        sample: { ok: true },
      },
    },
  },
};
