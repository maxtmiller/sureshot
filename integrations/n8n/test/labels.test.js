const { test } = require('node:test');
const assert = require('node:assert');
const { labels } = require('../dist/nodes/Sureshot/labels.js');
const { Sureshot } = require('../dist/nodes/Sureshot/Sureshot.node.js');
const { SureshotApi } = require('../dist/credentials/SureshotApi.credentials.js');

test('labels typed in a form get the types sureshot wants', () => {
	assert.deepStrictEqual(labels([{ question: 'team', answer: 'billing' }, { question: 'urgent', answer: 'True' }, { question: 'priority', answer: '2' }]),
		{ team: 'billing', urgent: true, priority: 2 });
});

test('the node and credential load', () => {
	const node = new Sureshot().description;
	assert.strictEqual(node.name, 'sureshot');
	assert.deepStrictEqual(node.properties[0].options.map((o) => o.value), ['decide', 'feedback']);
	assert.strictEqual(new SureshotApi().name, node.credentials[0].name);
});
