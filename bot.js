const fs = require("fs");
const { Client, LocalAuth } = require("whatsapp-web.js");
const qrcode = require("qrcode-terminal");
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

const client = new Client({
  authStrategy: new LocalAuth({ dataPath: process.env.AUTH_DIR || undefined }),
  puppeteer: {
    args: [
      "--no-sandbox",
      "--disable-dev-shm-usage",
      "--disable-gpu",
    ],
  },
});

let busy = false;
let me = null;
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

client.on("qr", (qr) => {
  latestQr = qr;
  console.log("[auth] Scan this QR in WhatsApp: Settings > Linked Devices > Link a Device");
  qrcode.generate(qr, { small: true });
});

client.on("ready", async () => {
  me = (client.info && client.info.wid && client.info.wid._serialized) || null;
  console.log(`[ready] WhatsApp connected as ${me || "unknown"}.`);
  const listChats = async (attempt) => {
    try {
      const chats = await client.getChats();
      for (const c of chats) {
        if (c.isGroup) console.log(`[group] ${c.name} | ${c.id._serialized}`);
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
});

const handledIds = new Set();

// reply: 남의 메시지는 msg.reply(인용답장), 본인 메시지는 그룹 ID로 직접 전송 —
// wwebjs는 fromMe 메시지의 from/to 의미가 반대라 msg.reply 의 경로가 달라질 수 있어
// 셀프 경로는 chatId 를 명시해 보낸다.
async function processAsk(msg, opts = {}) {
  const reply = opts.reply || ((text) => msg.reply(text));
  const id = msg.id && msg.id._serialized;
  if (!id) {
    console.log("[ask:no-id]");
    return;
  }
  if (handledIds.has(id)) {
    console.log("[ask:duplicate]");
    return;
  }
  handledIds.add(id);
  if (handledIds.size > 500) handledIds.clear();

  const triggered = extractTriggeredQuestion(msg, me);
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

  if (!allowOnce(msg.from)) {
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

client.on("message", async (msg) => {
  try {
    if (!GROUP_IDS.includes(msg.from)) {
      const body0 = (msg.body || "").trim();
      if (body0) {
        const chatId = [msg.from, msg.to].find(
          (j) => typeof j === "string" && j.endsWith("@g.us")
        );
        if (chatId && !GROUP_DISCOVERED.has(chatId)) {
          GROUP_DISCOVERED.add(chatId);
          console.log(`[discover] group not in GROUP_IDS: ${chatId}`);
        }
      }
      return;
    }
    if (msg.fromMe) return;
    const body = (msg.body || "").trim();
    if (!body) return;
    await processAsk(msg);
  } catch (e) {
    console.error("[err]", e);
  }
});

// 계정주 본인이 폰에서 직접 친 !ask에도 답한다. message 이벤트는 타인 메시지만
// 전달하는 경우가 있어 message_create에서 따로 받고, !ask 접두어로 한정해
// 봇 자신의 답변이 다시 트리거되는 것(무한 루프)을 막는다.
client.on("message_create", async (msg) => {
  try {
    if (!msg.fromMe) return;
    const chatId = [msg.from, msg.to].find(
      (j) => typeof j === "string" && j.endsWith("@g.us")
    );
    if (!chatId || !GROUP_IDS.includes(chatId)) return;
    const body = (msg.body || "").trim();
    if (!/^!ask\b/i.test(body)) return;
    console.log(`[self] owner !ask from=${msg.from} to=${msg.to}`);
    await processAsk(msg, { reply: (text) => client.sendMessage(chatId, text) });
  } catch (e) {
    console.error("[err]", e);
  }
});

client.initialize();
