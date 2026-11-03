import type { IExecuteSingleFunctions, IHttpRequestOptions, INodeType, INodeTypeDescription } from 'n8n-workflow';
import { NodeConnectionTypes } from 'n8n-workflow';
import { labels } from './labels';

export class Sureshot implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'sureshot',
		name: 'sureshot',
		icon: { light: 'file:sureshot.svg', dark: 'file:sureshot.dark.svg' },
		group: ['transform'],
		version: 1,
		subtitle: '={{$parameter["operation"]}}',
		description: 'Calibrated decisions from a fast model, escalated to a stronger one when it is unsure',
		defaults: { name: 'sureshot' },
		usableAsTool: true,
		inputs: [NodeConnectionTypes.Main],
		outputs: [NodeConnectionTypes.Main],
		credentials: [{ name: 'sureshotApi', required: true }],
		requestDefaults: {
			baseURL: '={{$credentials.url.replace(/\\/$/, "")}}',
			headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
		},
		properties: [
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				noDataExpression: true,
				default: 'decide',
				options: [
					{
						name: 'Decide',
						value: 'decide',
						action: 'Answer a question set about a text',
						description: 'Answer a question set about a text; the result has the decision ID to label later',
						routing: { request: { method: 'POST', url: '/decide' } },
					},
					{
						name: 'Feedback',
						value: 'feedback',
						action: 'Label a decision with its true answers',
						description: "Save a decision's true answers, which calibration measures the models against",
						routing: { request: { method: 'POST', url: '/feedback' } },
					},
				],
			},
			{
				displayName: 'Question Set Name or ID',
				name: 'questionSet',
				type: 'options',
				default: '',
				required: true,
				description: 'A set from the server\'s questions file. Choose from the list, or specify an ID using an <a href="https://docs.n8n.io/code/expressions/">expression</a>.',
				displayOptions: { show: { operation: ['decide'] } },
				typeOptions: {
					loadOptions: {
						routing: {
							request: { method: 'GET', url: '/questions' },
							output: {
								postReceive: [
									{ type: 'rootProperty', properties: { property: 'sets' } },
									{ type: 'setKeyValue', properties: { name: '={{$responseItem.name}}', value: '={{$responseItem.name}}' } },
									{ type: 'sort', properties: { key: 'name' } },
								],
							},
						},
					},
				},
				routing: { send: { type: 'body', property: 'questions' } },
			},
			{
				displayName: 'Text',
				name: 'text',
				type: 'string',
				typeOptions: { rows: 4 },
				default: '',
				required: true,
				description: 'What to decide about: an email, a ticket, a message',
				displayOptions: { show: { operation: ['decide'] } },
				routing: { send: { type: 'body', property: 'state' } },
			},
			{
				displayName: 'Options',
				name: 'options',
				type: 'collection',
				placeholder: 'Add Option',
				default: {},
				displayOptions: { show: { operation: ['decide'] } },
				options: [
					{
						displayName: 'Reference',
						name: 'ref',
						type: 'string',
						default: '',
						description: 'Your own ID for the item (a ticket or message ID), so a later label can find the decision',
						routing: { send: { type: 'body', property: 'ref' } },
					},
					{
						displayName: 'Escalate',
						name: 'escalate',
						type: 'boolean',
						default: true,
						description: 'Whether to ask the stronger model about what the fast one is unsure of',
						routing: { send: { type: 'body', property: 'escalate' } },
					},
				],
			},
			{
				displayName: 'Decision ID',
				name: 'decisionId',
				type: 'string',
				default: '',
				required: true,
				description: 'The ID a Decide step returned',
				displayOptions: { show: { operation: ['feedback'] } },
				routing: { send: { type: 'body', property: 'id' } },
			},
			{
				displayName: 'Labels',
				name: 'labels',
				type: 'fixedCollection',
				typeOptions: { multipleValues: true },
				default: {},
				placeholder: 'Add Label',
				description: "The true answers: a choice's key, a score level's 0-based position, or true/false",
				displayOptions: { show: { operation: ['feedback'] } },
				options: [
					{
						displayName: 'Label',
						name: 'label',
						values: [
							{ displayName: 'Question', name: 'question', type: 'string', default: '', description: 'The question\'s key, such as team' },
							{ displayName: 'Answer', name: 'answer', type: 'string', default: '' },
						],
					},
				],
				routing: {
					send: {
						preSend: [
							async function (this: IExecuteSingleFunctions, requestOptions: IHttpRequestOptions): Promise<IHttpRequestOptions> {
								const pairs = (this.getNodeParameter('labels.label', []) as Array<{ question: string; answer: string }>);
								requestOptions.body = { ...(requestOptions.body as object), labels: labels(pairs) };
								return requestOptions;
							},
						],
					},
				},
			},
		],
	};
}
