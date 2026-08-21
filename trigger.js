const USAGE = "Type \"!ask <your question>\" or @mention me with your question.";

function extractTriggeredQuestion(msg, me) {
  const body = (msg.body || "").trim();
  if (!body) return null;

  const prefixMatch = body.match(/^!ask\b/i);
  if (prefixMatch) {
    return { question: body.slice(prefixMatch[0].length).trim() };
  }

  const mentioned =
    Array.isArray(msg.mentionedIds) && !!me && msg.mentionedIds.includes(me);
  if (mentioned) {
    return { question: body.replace(/@\S+/g, "").trim() };
  }

  return null;
}

module.exports = { extractTriggeredQuestion, USAGE };
