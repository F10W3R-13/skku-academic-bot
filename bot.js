const fs = require("fs");
const { Client, LocalAuth } = require("whatsapp-web.js");
const qrcode = require("qrcode-terminal");
const { extractTriggeredQuestion, USAGE } = require("./trigger");

const config = JSON.parse(fs.readFileSync("./config.json", "utf8"));

const GROUP_IDS = process.env.GROUP_IDS
  ? process.env.GROUP_IDS.split(",").map(s => s.trim()).filter(Boolean)
  : config.groupIds;
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
  console.log("[auth] Scan this QR in WhatsApp: Settings > Linked Devices > Link a Device");
  qrcode.generate(qr, { small: true });
});

client.on("ready", async () => {
  me = (client.info && client.info.wid && client.info.wid._serialized) || null;
  console.log(`[ready] WhatsApp connected as ${me || "unknown"}.`);
  try {
    const chats = await client.getChats();
    for (const c of chats) {
      if (c.isGroup) console.log(`[group] ${c.name} | ${c.id._serialized}`);
    }
  } catch (e) {
    console.error("[ready] could not list chats:", e);
  }
});

client.on("message", async (msg) => {
  try {
    if (msg.fromMe) return;
    if (!GROUP_IDS.includes(msg.from)) return;
    const body = (msg.body || "").trim();
    if (!body) return;

    const triggered = extractTriggeredQuestion(msg, me);
    if (!triggered) {
      console.log("[skip] no trigger");
      return;
    }
    if (!triggered.question) {
      await msg.reply(USAGE);
      return;
    }
    const question = triggered.question;

    if (busy) {
      await msg.reply("One moment please — I answer one question at a time. 🙏");
      return;
    }
    busy = true;
    console.log(`[q] ${question}`);
    let data;
    try {
      data = await askApi(question);
    } catch (e) {
      console.error("[err] API:", e.message);
      await msg.reply("Sorry, I'm having trouble right now. Please try again in a moment.");
      return;
    } finally {
      busy = false;
    }

    const sources = Array.isArray(data.sources) && data.sources.length
      ? `\n\n📚 Source: ${data.sources.join(", ")}`
      : "";
    await msg.reply(`${data.answer}${sources}`);
    console.log("[a] replied.");
  } catch (e) {
    console.error("[err]", e);
  }
});

client.initialize();
