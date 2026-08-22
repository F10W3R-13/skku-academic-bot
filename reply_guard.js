// 답변 전송 전 마지막 가드레일.
//   - buildReply: LLM 출력을 그대로 그룹에 뿌리기 전에 이상 출력(탈옥 스팸 등)을
//     걸레말에서 잘라낸다. 상한은 정상 답변이 절대 못 미치는 값(MAX_ANSWER_TOKENS
//     2000 이면 영문 수천 자)이라 정상 답변은 잘리지 않는다.
//   - makeCooldown: 같은 방의 연타 !ask 로 토큰 비용을 낭비하는 걸 막는다.
const MAX_REPLY_CHARS = Number(process.env.MAX_REPLY_CHARS || 6000);
const COOLDOWN_MS = Number(process.env.COOLDOWN_MS || 10000);

function buildReply(data) {
  const sources =
    Array.isArray(data.sources) && data.sources.length
      ? `\n\n📚 Source: ${data.sources.join(", ")}`
      : "";
  let reply = `${(data && data.answer) || ""}${sources}`.trim();
  if (!reply) {
    return "Sorry, I couldn't produce an answer this time. Please try again in a moment.";
  }
  if (reply.length > MAX_REPLY_CHARS) {
    reply = reply.slice(0, MAX_REPLY_CHARS) + "\n\n…";
  }
  return reply;
}

function makeCooldown() {
  const lastAnswerAt = new Map();
  return function allow(chatId) {
    const now = Date.now();
    const prev = lastAnswerAt.get(chatId) || 0;
    if (now - prev < COOLDOWN_MS) return false;
    lastAnswerAt.set(chatId, now);
    return true;
  };
}

module.exports = { buildReply, makeCooldown, MAX_REPLY_CHARS, COOLDOWN_MS };
