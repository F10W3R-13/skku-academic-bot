// P1 스파이크: Baileys 실측 검증용 (bot.js 재작성 전 사전 확인).
// 목적: (a) 그룹 수신 (b) !ask 인용답장 (c) fromMe sendMessage
//       (d) 재시작 후 QR 없이 부팅 (e) RSS < 150MB
// 실행: node spike_baileys.js  (AUTH_DIR 환경변수로 세션 저장 위치 지정 가능)
// 종료 후 판정: 콘솔에 [spike] 라인으로 결과 출력.
const fs = require("fs");
const path = require("path");
const makeWASocket = require("baileys").default;
const {
  useMultiFileAuthState,
  fetchLatestBaileysVersion,
  DisconnectReason,
  Browsers,
} = require("baileys");
const qrcode = require("qrcode-terminal");

const AUTH_DIR = process.env.AUTH_DIR || ".spike-auth";
const API_URL = "http://localhost:8765/ask";

// 안전 게이트 (기본값 = 무응답):
//   SPIKE_OBSERVE=1      — 수신 로그만, 발신 경로 자체를 안 탄다(그룹 특정용).
//   SPIKE_ALLOW_JID=jid  — 이 그룹의 fromMe(소유자) 메시지에만 응답(일반 전송).
//   SPIKE_ALLOW_DM=jid   — 이 상대와의 DM에서 남의 메시지에 응답(인용답장 검증용).
//                          Railway 봇은 DM을 무시하므로 이중답변 없음 — (b)의 청정 시험장.
//   둘 다 없으면 어떤 메시지에도 답하지 않는다 — 실전 그룹 오염 방지(실측 교훈).
const OBSERVE = process.env.SPIKE_OBSERVE === "1";
const ALLOW_JID = (process.env.SPIKE_ALLOW_JID || "").trim() || null;
const ALLOW_DM = (process.env.SPIKE_ALLOW_DM || "").trim() || null;
// 자기채팅(나와의 채팅) 허용. fromMe 메시지는 봇 계정의 연결기기에서만 발생하므로
// 소유자 입력이 확정된다 — 그룹이 아닌 한 이 경로로 들어올 수 있는 사람이 없다.
const ALLOW_SELF = process.env.SPIKE_ALLOW_SELF === "1";

let latestQr = null;

// 로컬 스파이크용: 진짜 RAG API 없이 발신 경로(b)/(c)를 검증하기 위한 모킹.
// SPIKE_MOCK=1 이면 8765 에 가짜 /ask 를 띄운다.
if (process.env.SPIKE_MOCK === "1") {
  const http = require("http");
  http
    .createServer((req, res) => {
      if (req.url === "/ask" && req.method === "POST") {
        let raw = "";
        req.on("data", (c) => (raw += c));
        req.on("end", () => {
          let q = "";
          try { q = JSON.parse(raw).question || ""; } catch {}
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({
            answer: `스파이크 모킹 답변 — 받은 질문: "${q.slice(0, 80)}"`,
            sources: ["spike-mock"],
          }));
        });
      } else {
        res.writeHead(404).end();
      }
    })
    .listen(8765, () => console.log("[spike] mock /ask on :8765"));
}

// 운영과 동일한 QR 웹페이지 (PORT+QR_TOKEN 세팅 시). 로컬: http://localhost:<PORT>/?t=<QR_TOKEN>
if (process.env.PORT && process.env.QR_TOKEN) {
  const { startQrServer } = require("./qr_server");
  startQrServer({
    port: Number(process.env.PORT),
    getToken: () => process.env.QR_TOKEN,
    getQr: () => latestQr,
  }).then(() => console.log("[spike] QR page ready"));
}

function rssMB() {
  return Math.round(process.memoryUsage().rss / 1024 / 1024);
}

function extractBody(message) {
  if (!message) return "";
  return (
    message.conversation ||
    (message.extendedTextMessage && message.extendedTextMessage.text) ||
    (message.imageMessage && message.imageMessage.caption) ||
    ""
  );
}

async function askApi(question) {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal: AbortSignal.timeout(60000),
  });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

async function main() {
  const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR);
  const { version } = await fetchLatestBaileysVersion();
  console.log(`[spike] WA protocol version: ${version}`);
  console.log(`[spike] auth dir: ${path.resolve(AUTH_DIR)}`);
  console.log(`[spike] RSS at boot: ${rssMB()}MB`);

  const sock = makeWASocket({
    auth: state,
    version,
    printQRInTerminal: false,
    // 페어링코드/QR 모두 데스크톱 플랫폼 식별자가 안전하다(모바일 식별은 거절됨).
    browser: Browsers.ubuntu("Chrome"),
    // 히스토리 전체 동기화는 세션 부팅을 무겁게 만든다 — 이 봇은 신규 메시지만 필요.
    syncFullHistory: false,
  });

  // QR 대신 페어링 코드(전화번호 8자리)로 연결: 폰 카메라/스캔 타이밍 문제를 우회.
  // 사용: PAIR_PHONE=8210xxxxxxxx 환경변수 (국가번호 포함, + 나 - 없이 숫자만)
  // 중요: requestPairingCode 는 딱 1회만 — QR 로테이션마다 재요청하면 이전 코드가
  // 전부 무효화되어 '잘못된 코드'가 된다(실측).
  let pairingRequested = false;
  sock.ev.on("connection.update", (u) => {
    if (u.qr && process.env.PAIR_PHONE && !pairingRequested) {
      pairingRequested = true;
      const phone = process.env.PAIR_PHONE.replace(/[^0-9]/g, "");
      sock.requestPairingCode(phone)
        .then((code) => {
          const pretty = code.match(/.{1,4}/g).join("-");
          console.log(`\n[spike] PAIRING CODE: ${pretty}`);
          console.log("[spike] 폰 WhatsApp > 연결된 기기 > 기기 연결 > 대신 코드로 연결 에 입력");
        })
        .catch((e) => {
          console.error("[spike] pairing code failed:", e.message);
          pairingRequested = false; // 다음 QR에서 재시도 허용
        });
    }
  });

  sock.ev.on("creds.update", saveCreds);

  sock.ev.on("connection.update", (update) => {
    const { connection, lastDisconnect, qr } = update;
    if (qr) {
      latestQr = qr;
      console.log("[spike] QR arrived — scan with the BOT number's phone:");
      qrcode.generate(qr, { small: true });
    }
    if (connection === "open") {
      console.log(`[spike] CONNECTED as ${sock.user && sock.user.id}. RSS: ${rssMB()}MB`);
      if (!fs.existsSync(path.join(AUTH_DIR, "creds.json"))) {
        console.log("[spike] WARN: connected but creds.json missing?");
      } else {
        console.log("[spike] (d) creds.json persisted — restart me to verify QR-free boot");
      }
      setInterval(() => console.log(`[spike] heartbeat RSS: ${rssMB()}MB`), 30000);
    }
    if (connection === "close") {
      const statusCode = lastDisconnect && lastDisconnect.error
        ? lastDisconnect.error.output && lastDisconnect.error.output.statusCode
        : null;
      const reason = DisconnectReason[statusCode] || `code ${statusCode}`;
      console.log(`[spike] closed: ${reason}`);
      if (statusCode === DisconnectReason.loggedOut) {
        console.log("[spike] logged out — delete auth dir and rescan");
        process.exit(1);
      }
      console.log("[spike] reconnecting in 3s...");
      setTimeout(main, 3000);
    }
  });

  sock.ev.on("messages.upsert", async ({ messages, type }) => {
    for (const m of messages) {
      const jid = m.key && m.key.remoteJid;
      const body = extractBody(m.message).trim();
      const fromMe = !!(m.key && m.key.fromMe);
      console.log(
        `[spike] msg type=${type} jid=${jid} fromMe=${fromMe} id=${m.key && m.key.id} body=${body.slice(0, 60) || "(empty)"}`
      );
      if (OBSERVE) continue; // 관찰 모드: 절대 발신하지 않는다
      const allowGroup = ALLOW_JID && jid === ALLOW_JID && fromMe;
      const allowDm = ALLOW_DM && jid === ALLOW_DM && !fromMe;
      const allowSelf = ALLOW_SELF && fromMe && !(jid || "").endsWith("@g.us");
      if (!allowGroup && !allowDm && !allowSelf) continue; // 기본 무응답
      if (!body) continue;

      const isGroup = typeof jid === "string" && jid.endsWith("@g.us");

      // (a) 그룹 수신 확인은 로그로 충분. 아래는 (b)/(c) 발신 검증.
      const isAsk = /^!ask\b/i.test(body);
      if (!isAsk) continue;

      const question = body.replace(/^!ask\s*/i, "").trim();
      if (!question) {
        console.log("[spike] !ask with no question — skip");
        continue;
      }

      try {
        const data = await askApi(question);
        const text = `(spike) ${data.answer}`.slice(0, 1000);
        // 운영 bot.js와 동일 규칙: 남의 메시지=인용답장, 본인(fromMe)=일반 전송.
        // quoted 에는 메시지 객체 전체(WAMessage)를 넘긴다 — key만 넘기면
        // 내부에서 contextInfo.fromMe 를 읽지 못해 TypeError(실측).
        if (fromMe) {
          await sock.sendMessage(jid, { text });
        } else {
          await sock.sendMessage(jid, { text }, { quoted: m });
        }
        console.log(`[spike] replied (${fromMe ? "plain" : "quoted"}). RSS: ${rssMB()}MB`);
      } catch (e) {
        console.error(`[spike] reply failed: ${e.message}`);
      }
    }
  });

  sock.ev.on("group-participants.update", (u) => {
    console.log(`[spike] group-participants.update ${u.id} (${u.action}) — event plumbing works`);
  });
}

main().catch((e) => {
  console.error("[spike] fatal:", e);
  process.exit(1);
});
