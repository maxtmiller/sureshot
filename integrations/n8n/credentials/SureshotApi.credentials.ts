import type { IAuthenticateGeneric, ICredentialTestRequest, ICredentialType, Icon, INodeProperties } from 'n8n-workflow';

export class SureshotApi implements ICredentialType {
	name = 'sureshotApi';

	displayName = 'Sureshot API';

	icon: Icon = { light: 'file:../nodes/Sureshot/sureshot.svg', dark: 'file:../nodes/Sureshot/sureshot.dark.svg' };

	documentationUrl = 'https://github.com/maxtmiller/sureshot/blob/main/docs/guide.md#http-api-and-configuration';

	properties: INodeProperties[] = [
		{
			displayName: 'URL',
			name: 'url',
			type: 'string',
			default: '',
			placeholder: 'https://sureshot.example.com',
			description: 'Where sureshot serve runs',
			required: true,
		},
		{
			displayName: 'API Key',
			name: 'apiKey',
			type: 'string',
			typeOptions: { password: true },
			default: '',
			description: "The server's SURESHOT_API_KEY",
			required: true,
		},
	];

	authenticate: IAuthenticateGeneric = {
		type: 'generic',
		properties: { headers: { Authorization: '=Bearer {{$credentials.apiKey}}' } },
	};

	test: ICredentialTestRequest = {
		request: { baseURL: '={{$credentials.url.replace(/\\/$/, "")}}', url: '/questions' },
	};
}
