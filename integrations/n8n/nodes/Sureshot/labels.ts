/** A label typed in a form is text; sureshot wants true/false for yes-no questions and a number for a score's level. */
export function labelValue(answer: string): string | number | boolean {
	const text = answer.trim();
	if (/^(true|false)$/i.test(text)) return text.toLowerCase() === 'true';
	if (/^-?\d+$/.test(text)) return Number(text);
	return text;
}

export function labels(pairs: Array<{ question: string; answer: string }>): Record<string, string | number | boolean> {
	return Object.fromEntries(pairs.map(({ question, answer }) => [question, labelValue(answer)]));
}
