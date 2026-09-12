// adapter.js 유닛 테스트 — Baileys 페이로드 → 표준 메시지 변환 4케이스 + 보조함수.
// 실제 네트워크/소켓 없이 순수 함수만 검증한다(P1 에서 전송계층은 이미 실측됨).
const assert = require("assert");
const {
  extractBody,
  normalizeJid,
  extractMentionedIds,
  translateIncoming,
  meCandidates,
  classifyClose,
} = require("../adapter");

let passed = 0;
function check(name, fn) {
  try {
    fn();
    passed++;
    console.log(`ok: ${name}`);
  } catch (e) {
    console.error(`FAIL: ${name}`);
    console.error(e);
    process.exit(1);
  }
}

// 1) 일반 텍스트(conversation)
const convMsg = {
  key: { remoteJid: "120363428912675030@g.us", fromMe: false, id: "AAA111" },
  message: { conversation: "!ask 수강신청" },
};
// 2) 멘션 포함(extendedTextMessage + contextInfo.mentionedJid)
const mentionMsg = {
  key: { remoteJid: "120363428912675030@g.us", fromMe: false, id: "BBB222" },
  message: {
    extendedTextMessage: {
      text: "@821056233717 언제가 마감이야?",
      contextInfo: { mentionedJid: ["821056233717@s.whatsapp.net"] },
    },
  },
};
// 3) fromMe (봇 폰에서 직접 보낸 !ask)
const fromMeMsg = {
  key: { remoteJid: "120363428912675030@g.us", fromMe: true, id: "CCC333" },
  message: { conversation: "!ask 셀프 질문" },
};
// 4) 비그룹(개인 DM — 운영 봇은 무시해야 하는 경로)
const dmMsg = {
  key: { remoteJid: "8210551234567@s.whatsapp.net", fromMe: false, id: "DDD444" },
  message: { conversation: "안녕?" },
};

check("extractBody: conversation", () => {
  assert.strictEqual(extractBody(convMsg.message), "!ask 수강신청");
});
check("extractBody: extendedText(멘션)", () => {
  assert.strictEqual(extractBody(mentionMsg.message), "@821056233717 언제가 마감이야?");
});
check("extractBody: null 안전", () => {
  assert.strictEqual(extractBody(null), "");
  assert.strictEqual(extractBody({}), "");
});

check("normalizeJid: @s.whatsapp.net → @c.us", () => {
  assert.strictEqual(normalizeJid("821056233717@s.whatsapp.net"), "821056233717@c.us");
});
check("normalizeJid: device 접미사(:9) 제거", () => {
  assert.strictEqual(normalizeJid("821056233717:9@s.whatsapp.net"), "821056233717@c.us");
});
check("normalizeJid: @lid → 번호@c.us", () => {
  assert.strictEqual(normalizeJid("101507290644631@lid"), "101507290644631@c.us");
});
check("normalizeJid: null/이상값 안전", () => {
  assert.strictEqual(normalizeJid(null), null);
  assert.strictEqual(normalizeJid(undefined), null);
  assert.strictEqual(normalizeJid(42), null);
});

check("extractMentionedIds: 파싱+정규화", () => {
  assert.deepStrictEqual(extractMentionedIds(mentionMsg.message), ["821056233717@c.us"]);
});
check("extractMentionedIds: 멘션 없으면 빈 배열", () => {
  assert.deepStrictEqual(extractMentionedIds(convMsg.message), []);
});

check("translateIncoming: conversation 케이스", () => {
  const t = translateIncoming({ ...convMsg, messageTimestamp: 1726000000 });
  assert.strictEqual(t.id, "AAA111");
  assert.strictEqual(t.from, "120363428912675030@g.us");
  assert.strictEqual(t.body, "!ask 수강신청");
  assert.strictEqual(t.fromMe, false);
  assert.strictEqual(t.isGroup, true);
  assert.strictEqual(t.timestamp, 1726000000);
  assert.deepStrictEqual(t.mentionedIds, []);
});
check("translateIncoming: 멘션 케이스", () => {
  const t = translateIncoming(mentionMsg);
  assert.strictEqual(t.body, "@821056233717 언제가 마감이야?");
  assert.deepStrictEqual(t.mentionedIds, ["821056233717@c.us"]);
});
check("translateIncoming: fromMe 케이스", () => {
  const t = translateIncoming(fromMeMsg);
  assert.strictEqual(t.fromMe, true);
  assert.strictEqual(t.isGroup, true);
});
check("translateIncoming: DM 케이스", () => {
  const t = translateIncoming(dmMsg);
  assert.strictEqual(t.isGroup, false);
  assert.strictEqual(t.from, "8210551234567@s.whatsapp.net");
});
check("translateIncoming: key 누락 등 기형 페이로드 안전", () => {
  const t = translateIncoming(null);
  assert.strictEqual(t.id, null);
  assert.strictEqual(t.body, "");
  assert.strictEqual(t.fromMe, false);
});

check("meCandidates: PN+LID 후보 생성", () => {
  const cands = meCandidates({ id: "821056233717:9@s.whatsapp.net", lid: "101507290644631:9@lid" });
  assert.ok(cands.includes("821056233717@c.us"));
  assert.ok(cands.includes("101507290644631@c.us"));
});
check("meCandidates: null 안전", () => {
  assert.deepStrictEqual(meCandidates(null), []);
});

check("classifyClose: loggedOut = 재연결 금지", () => {
  const r = classifyClose({ error: { output: { statusCode: 401 } } });
  assert.strictEqual(r.isLoggedOut, true);
  assert.strictEqual(r.shouldReconnect, false);
});
check("classifyClose: 일반 끊김(428 등) = 재연결", () => {
  const r = classifyClose({ error: { output: { statusCode: 428 } } });
  assert.strictEqual(r.isLoggedOut, false);
  assert.strictEqual(r.shouldReconnect, true);
});
check("classifyClose: lastDisconnect 없음 = 재연결", () => {
  const r = classifyClose(undefined);
  assert.strictEqual(r.shouldReconnect, true);
});

console.log(`\n${passed} checks passed`);
