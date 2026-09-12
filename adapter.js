// Baileys ⇔ 표준 메시지 어댑터.
// 이 파일이 프로젝트에서 유일하게 baileys 를 import 하는 곳이다(설계 규칙).
// bot.js 는 여기 나오는 표준 메시지 모양만 본다 — wwebjs 시절 모양을 최대한
// 유지해서 trigger.js / reply_guard.js 를 무수정 재사용한다.
const {
  makeWASocket,
  useMultiFileAuthState,
  fetchLatestBaileysVersion,
  DisconnectReason,
  Browsers,
} = require("baileys");

// Baileys 메시지 본문 추출. 멘션 메시지는 extendedTextMessage 로 온다.
function extractBody(message) {
  if (!message) return "";
  return (
    message.conversation ||
    (message.extendedTextMessage && message.extendedTextMessage.text) ||
    (message.imageMessage && message.imageMessage.caption) ||
    (message.videoMessage && message.videoMessage.caption) ||
    ""
  );
}

// mentionedJid 는 "8210...@s.whatsapp.net" 또는 LID 마이그레이션 후 "1015...@lid"
// 형태. wwebjs 호환을 위해 번호 부분만 뽑아 "@c.us" 로 정규화한다.
function normalizeJid(jid) {
  if (typeof jid !== "string") return null;
  const num = jid.split("@")[0].split(":")[0];
  return num ? `${num}@c.us` : null;
}

function extractMentionedIds(message) {
  const ctx =
    message &&
    message.extendedTextMessage &&
    message.extendedTextMessage.contextInfo;
  const list = (ctx && ctx.mentionedJid) || [];
  return list.map(normalizeJid).filter(Boolean);
}

// 수신 Baileys 메시지(WAMessage) → 표준 메시지. bot.js 파이프라인의 입력 단위.
function translateIncoming(m) {
  const key = (m && m.key) || {};
  const jid = key.remoteJid || "";
  return {
    // wwebjs msg.id._serialized 상당 — 중복제거 키(S5).
    id: key.id || null,
    from: jid,
    to: null, // Baileys 수신 페이로드에 없음. 합성 키 폴백에서만 쓰던 값.
    body: extractBody(m && m.message),
    fromMe: !!key.fromMe,
    isGroup: jid.endsWith("@g.us"),
    mentionedIds: extractMentionedIds(m && m.message),
    // unix 초. 중복제거 합성 키(S5)와 로그에 쓴다 — wwebjs msg.timestamp 상당.
    timestamp: Number((m && m.messageTimestamp) || 0),
    raw: m, // 인용답장(quoted)에 필요한 원본
  };
}

// 소켓 유저 id("8210...:9@s.whatsapp.net") → "@c.us" 형태. trigger.js 의 me 와
// mentionedIds 비교는 이 형태로만 일치한다. LID 전용 멘션(@lid)은 두 번째 후보로.
function meCandidates(sockUser) {
  if (!sockUser) return [];
  const cands = [];
  const pn = normalizeJid(sockUser.id);
  if (pn) cands.push(pn);
  if (sockUser.lid) {
    const lidNum = String(sockUser.lid).split("@")[0].split(":")[0];
    if (lidNum) cands.push(`${lidNum}@c.us`);
  }
  return cands;
}

// 재연결 정책(S12). loggedOut(세션 폭사)만 재인증 필요 — 나머지는 백오프 재연결.
function classifyClose(lastDisconnect) {
  const err = lastDisconnect && lastDisconnect.error;
  const statusCode =
    (err && err.output && err.output.statusCode) ||
    (err && err.isBoom && err.output && err.output.statusCode) ||
    null;
  return {
    statusCode,
    isLoggedOut: statusCode === DisconnectReason.loggedOut,
    shouldReconnect: statusCode !== DisconnectReason.loggedOut,
  };
}

module.exports = {
  makeWASocket,
  useMultiFileAuthState,
  fetchLatestBaileysVersion,
  Browsers,
  DisconnectReason,
  extractBody,
  normalizeJid,
  extractMentionedIds,
  translateIncoming,
  meCandidates,
  classifyClose,
};
