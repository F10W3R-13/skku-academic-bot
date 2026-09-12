// SKKU 규정 봇 — Baileys 버전.
// WhatsApp 연결은 adapter.js(baileys 유일 의존점)에 맡기고, 이 파일은
// 트리거 → 쿨다운 → RAG API → 가드 파이프라인만 책임진다.
// 행동 계약(S1~S12)은 docs/plans/2026-09-12-baileys-migration.md 참조.
const fs = require("fs");
const qrcode = require("qrcode-terminal");
const {
  makeWASocket,
  useMultiFileAuthState,
  fetchLatestBaileysVersion,
  Browsers,
  DisconnectReason,
  translateIncoming,
  meCandidates,
  classifyClose,
} = require("./adapter");
const { extractTriggeredQuestion, USAGE } = require("./trigger");
const { buildReply, makeCooldown, COOLDOWN_MS } = require("./reply_guard");

const allowOnce = makeCooldown();

const config = JSON.parse(fs.readFileSync("./config.json", "utf8"));

const GROUP_IDS = process.env.GROUP_IDS
  ? process.env.GROUP_IDS.split(",").map(s => s.trim()).filter(Boolean)
  : config.groupIds;
const GROUP_DISCOVERED = new Set();
if (!Array.isArray(GROUP_IDS) || GROUP_IDS.length === 0) {
  console.log("[setup] No target groups configured.");
  console.log("[setup] Set GROUP_IDS env var (comma-separated) or config.json groupIds, using the [group] IDs logged below.");
  console.log("[setup] After startup, copy your group's ID below and restart.");
}

const AUTH_DIR = process.env.AUTH_DIR || undefined;
// Baileys 세션은 AUTH_DIR 하위 전용 폴더로 격리한다. AUTH_DIR(/data) 루트에는
// index.json·logs 등이 살아 있고, loggedOut 시 세션 폴더만 삭제할 수 있어야 한다.
const BAILEYS_AUTH_DIR =
  process.env.BAILEYS_AUTH_DIR ||
  (AUTH_DIR ? require("path").join(AUTH_DIR, "baileys-auth") : "baileys_auth");
// QR 스캔이 어려운 환경용 대체 인증: PAIR_PHONE 에 번호(국가번호 포함, 숫자만)를
// 넣으면 QR 대신 8자리 페어링 코드를 로그에 뽑는다(폰에서 '대신 코드로 연결').
// 실측: requestPairingCode 는 QR 로테이션과 무관하게 세션당 1회만 호출해야 한다 —
// 재호출하면 직전 코드가 무효화되고 세션이 loggedOut 으로 붕괴한다.
const PAIR_PHONE = (process.env.PAIR_PHONE || "").replace(/[^0-9]/g, "") || null;

let busy = false;
let me = null; // "@c.us" 형태 후보들 — trigger.js 멘션 비교용
let latestQr = null;

const { startQrServer } = require("./qr_server");
if (process.env.PORT && process.env.QR_TOKEN) {
  startQrServer({
    port: Number(process.env.PORT),
    getToken: () => process.env.QR_TOKEN,
    getQr: () => latestQr,
  }).then(() => console.log(`[qr] QR page available at /?t=<QR_TOKEN>`));
} else if (process.env.PORT && !process.env.QR_TOKEN) {
  console.log("[qr] QR_TOKEN not set — QR web page disabled");
}

async function askApi(question) {
  const res = await fetch(`${config.apiUrl}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal: AbortSignal.timeout(60000),
  });
  if (!res.ok) throw new Error(`API responded ${res.status}`);
  return res.json();
}

// 발신: 남의 메시지는 인용답장(S6), 본인(fromMe) 메시지는 그룹 ID로 직접 전송 —
// wwebjs 시절과 동일 규칙. quoted 에는 WAMessage 전체를 넘겨야 한다(key만 넘기면
// TypeError: reading 'fromMe' — P1 실측).
function makeReply(sock, t) {
  if (t.fromMe) {
    return (text) => sock.sendMessage(t.from, { text });
  }
  return (text) => sock.sendMessage(t.from, { text }, { quoted: t.raw });
}

// S1~S11 파이프라인 — wwebjs bot.js 에서 로직 무이식(동일 순서/동일 문구).
async function processAsk(sock, t, opts = {}) {
  const reply = opts.reply || makeReply(sock, t);
  // LID 주소 체계의 페이로드는 msg.id 가 비어있는 경우가 있어 합성 키로
  // 중복제거를 대체한다(message_create/ack 이중 수신도 같은 키로 잡힘).
  const id =
    t.id ||
    `${t.from}|${t.to}|${t.timestamp}|${(t.body || "").slice(0, 80)}`;
  if (handledIds.has(id)) {
    console.log("[ask:duplicate]");
    return;
  }
  handledIds.add(id);
  if (handledIds.size > 500) handledIds.clear();

  const triggered = extractTriggeredQuestion(t, me);
  if (!triggered) {
    console.log("[ask:no-trigger]");
    return;
  }
  if (!triggered.question) {
    console.log("[ask:usage]");
    await reply(USAGE);
    return;
  }
  const question = triggered.question;

  if (!allowOnce(t.from)) {
    console.log("[ask:cooldown]");
    await reply(
      `One moment please — I can take one question every ${Math.round(COOLDOWN_MS / 1000)}s. 🙏`
    );
    return;
  }
  if (busy) {
    console.log("[ask:busy]");
    await reply("One moment please — I answer one question at a time. 🙏");
    return;
  }
  busy = true;
  console.log(`[q] ${question}`);
  let data;
  try {
    data = await askApi(question);
  } catch (e) {
    console.error("[err] API:", e.message);
    await reply("Sorry, I'm having trouble right now. Please try again in a moment.");
    return;
  } finally {
    busy = false;
  }

  await reply(buildReply(data));
  console.log("[a] replied.");
}

const handledIds = new Set();

// S12 재연결: loggedOut(세션 폭사)만 creds 삭제+재QR, 나머지는 백오프 재연결.
// 최근 실패 누적 시 대기를 늘려 무한 크래시 루프를 막는다.
let reconnectDelays = [3000, 6000, 12000, 30000, 60000];
let pairingRequested = false;

async function start() {
  const { state, saveCreds } = await useMultiFileAuthState(BAILEYS_AUTH_DIR);
  const { version } = await fetchLatestBaileysVersion().catch(() => ({ version: undefined }));
  const sock = makeWASocket({
    auth: state,
    version,
    printQRInTerminal: false,
    browser: Browsers.ubuntu("Chrome"),
    syncFullHistory: false,
  });

  sock.ev.on("creds.update", saveCreds);

  sock.ev.on("connection.update", async (update) => {
    const { connection, lastDisconnect, qr } = update;
    if (qr) {
      latestQr = qr;
      if (PAIR_PHONE && !pairingRequested) {
        pairingRequested = true;
        sock
          .requestPairingCode(PAIR_PHONE)
          .then((code) => {
            const pretty = code.match(/.{1,4}/g).join("-");
            console.log(`[auth] PAIRING CODE: ${pretty}`);
            console.log("[auth] 폰 WhatsApp > 연결된 기기 > 기기 연결 > 대신 코드로 연결");
          })
          .catch((e) => {
            console.error("[auth] pairing code failed:", e.message);
            pairingRequested = false;
          });
      } else {
        console.log("[auth] Scan this QR in WhatsApp: Settings > Linked Devices > Link a Device");
        qrcode.generate(qr, { small: true });
      }
    }
    if (connection === "open") {
      reconnectDelays = [3000, 6000, 12000, 30000, 60000];
      me = meCandidates(sock.user);
      console.log(`[ready] WhatsApp connected as ${(me && me[0]) || "unknown"}.`);
      // S1 그룹 로깅 — 부팅 직후 getChats 가 실패하는 일이 있어 재시도(wwebjs 동작 유지).
      const listChats = async (attempt) => {
        try {
          const groups = await sock.groupFetchAllParticipating();
          for (const gid of Object.keys(groups)) {
            console.log(`[group] ${groups[gid].subject} | ${gid}`);
          }
        } catch (e) {
          if (attempt < 3) {
            console.log(`[ready] chat list not ready, retrying in 10s (${attempt}/3)...`);
            setTimeout(() => listChats(attempt + 1), 10000);
          } else {
            console.error("[ready] could not list chats after retries — use [discover] lines instead");
          }
        }
      };
      listChats(1);
    }
    if (connection === "close") {
      const { isLoggedOut, shouldReconnect } = classifyClose(lastDisconnect);
      if (!shouldReconnect) {
        console.error("[auth] logged out — session invalid, deleting credentials for re-pair");
        try {
          fs.rmSync(BAILEYS_AUTH_DIR, { recursive: true, force: true });
        } catch {}
        process.exit(1); // Railway 재시작 → QR 재인증 (qr_server 이용)
      }
      const delay = reconnectDelays.shift() || 60000;
      console.log(`[conn] closed — reconnecting in ${Math.round(delay / 1000)}s...`);
      setTimeout(start, delay);
    }
  });

  // wwebjs 'message'+'message_create' 를 messages.upsert 하나로 통합 처리한다.
  // fromMe 분기는 processAsk/trigger 내부에서 이미 하던 일이라 표면은 동일하다.
  sock.ev.on("messages.upsert", async ({ messages, type }) => {
    // 'append' 는 부팅 직후의 히스토리/오프라인 동기화다 — 옛날 !ask 에 늦답변하는
    // 사고를 막기 위해 신규 메시지(notify)만 처리한다(wwebjs 대비 안전 강화).
    if (type && type !== "notify") return;
    for (const m of messages) {
      const t = translateIncoming(m);
      try {
        // 미등록 그룹 발견 로깅(S8) — fromMe 무관.
        if (t.isGroup && !GROUP_IDS.includes(t.from) && t.body.trim()) {
          if (!GROUP_DISCOVERED.has(t.from)) {
            GROUP_DISCOVERED.add(t.from);
            console.log(`[discover] group not in GROUP_IDS: ${t.from}`);
          }
          continue;
        }
        if (!t.isGroup) continue; // 개인 DM 무시 — 운영 범위는 그룹뿐(S1).
        if (t.fromMe) {
          // 본인(폰)이 직접 친 !ask(S7). 접두어 한정은 trigger.js 가 한다.
          if (!/^!ask\b/i.test(t.body.trim())) continue;
          console.log(`[self] owner !ask from=${t.from}`);
          await processAsk(sock, t);
          continue;
        }
        const body = t.body.trim();
        if (!body) continue;
        await processAsk(sock, t);
      } catch (e) {
        console.error("[err]", e);
      }
    }
  });

  return sock;
}

start().catch((e) => {
  console.error("[fatal]", e);
  process.exit(1);
});
