const assert = require("assert");

process.env.MAX_REPLY_CHARS = "100";
process.env.COOLDOWN_MS = "30";
const { buildReply, makeCooldown } = require("../reply_guard");

let passed = 0;
function check(name, fn) {
  try {
    fn();
    passed++;
    console.log(`ok: ${name}`);
  } catch (e) {
    console.error(`FAIL: ${name}`);
    console.error(e.message);
    process.exit(1);
  }
}

check("normal answer passes through with sources", () => {
  const r = buildReply({ answer: "You need 140 credits.", sources: ["졸업요건"] });
  assert.strictEqual(r, "You need 140 credits.\n\n📚 Source: 졸업요건");
});

check("oversized answer is truncated at the ceiling", () => {
  const r = buildReply({ answer: "가".repeat(500), sources: [] });
  assert.strictEqual(r.length, 100 + "\n\n…".length);
  assert.ok(r.endsWith("…"));
});

check("empty answer falls back to an apology, never an empty message", () => {
  assert.ok(buildReply({ answer: "", sources: [] }).length > 0);
  assert.ok(buildReply({}).length > 0);
});

check("cooldown allows first ask, blocks immediate second, allows after window", async () => {
  const allow = makeCooldown();
  assert.strictEqual(allow("room@g.us"), true);
  assert.strictEqual(allow("room@g.us"), false, "쿨다운 내 재요청은 차단");
  await new Promise((res) => setTimeout(res, 40));
  assert.strictEqual(allow("room@g.us"), true, "쿨다운 지나면 다시 허용");
});

check("cooldown is per chat, not global", () => {
  const allow = makeCooldown();
  assert.strictEqual(allow("a@g.us"), true);
  assert.strictEqual(allow("b@g.us"), true, "다른 방은 영향 없음");
});

console.log(`\n${passed} checks passed`);
