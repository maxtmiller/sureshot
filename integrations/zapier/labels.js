// A label typed in a Zap is text; sureshot wants true/false for yes-no questions and a number for a score's level.
const labelValue = (answer) => {
  const text = String(answer).trim();
  if (/^(true|false)$/i.test(text)) return text.toLowerCase() === 'true';
  if (/^-?\d+$/.test(text)) return Number(text);
  return text;
};

module.exports = { labelValue };
