const assert = require("assert");
const { extractTriggeredQuestion, USAGE } = require("../trigger");

const BOT = "821012345678@c.us";

let passed = 0;
function check(name, fn) {
  try {
    fn();
    passed++;
  } catch (e) {
    console.error(`FAIL: ${name}`);
    console.error(e.message);
    process.exit(1);
  }
}

check("!ask prefix (lowercase) extracts question", () => {
  const r = extractTriggeredQuestion({ body: "!ask What is the tuition deadline?" }, null);
  assert.deepStrictEqual(r, { question: "What is the tuition deadline?" });
});

check("!ask prefix is case-insensitive", () => {
  const r = extractTriggeredQuestion({ body: "!AsK how many credits per semester?" }, null);
  assert.strictEqual(r.question, "how many credits per semester?");
});

check("!ask prefix requires word boundary", () => {
  assert.strictEqual(extractTriggeredQuestion({ body: "!askme hi" }, null), null);
  assert.strictEqual(extractTriggeredQuestion({ body: "please!ask hi" }, null), null);
});

check("bare !ask yields empty question", () => {
  assert.deepStrictEqual(
    extractTriggeredQuestion({ body: "!ask" }, null),
    { question: "" }
  );
  assert.deepStrictEqual(
    extractTriggeredQuestion({ body: "  !ASK   " }, null),
    { question: "" }
  );
});

check("mention by bot triggers and strips @tags", () => {
  const msg = {
    body: "@821012345678@c.us what about tuition?",
    mentionedIds: [BOT],
  };
  const r = extractTriggeredQuestion(msg, BOT);
  assert.strictEqual(r.question, "what about tuition?");
});

check("mention strips multiple @tags", () => {
  const msg = {
    body: "@821012345678@c.us @someoneelse when is add/drop?",
    mentionedIds: [BOT, "999888777@c.us"],
  };
  const r = extractTriggeredQuestion(msg, BOT);
  assert.strictEqual(r.question, "when is add/drop?");
});

check("mention with no text left after stripping yields empty question", () => {
  const msg = { body: "@821012345678@c.us", mentionedIds: [BOT] };
  assert.deepStrictEqual(extractTriggeredQuestion(msg, BOT), { question: "" });
});

check("mention without me set does not trigger", () => {
  const msg = {
    body: "@821012345678@c.us hello",
    mentionedIds: [BOT],
  };
  assert.strictEqual(extractTriggeredQuestion(msg, null), null);
  assert.strictEqual(extractTriggeredQuestion(msg, undefined), null);
});

check("mention of someone else does not trigger", () => {
  const msg = {
    body: "@999888777@c.us are you the bot?",
    mentionedIds: ["999888777@c.us"],
  };
  assert.strictEqual(extractTriggeredQuestion(msg, BOT), null);
});

check("non-triggered chatter returns null", () => {
  assert.strictEqual(
    extractTriggeredQuestion({ body: "hi everyone, how was class?" }, BOT),
    null
  );
  assert.strictEqual(extractTriggeredQuestion({ body: "" }, BOT), null);
  assert.strictEqual(extractTriggeredQuestion({}, BOT), null);
});

check("usage string matches spec", () => {
  assert.strictEqual(USAGE, 'Type "!ask <your question>" or @mention me with your question.');
});

console.log(`${passed} checks passed`);
console.log("ALL PASS");
