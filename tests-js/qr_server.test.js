const assert = require("assert");
const { startQrServer } = require("../qr_server");

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

async function checkAsync(name, fn) {
  try {
    await fn();
    passed++;
  } catch (e) {
    console.error(`FAIL: ${name}`);
    console.error(e.message);
    process.exit(1);
  }
}

(async () => {
  let qr = null;
  const server = await startQrServer({
    port: 0,
    getToken: () => "secret123",
    getQr: () => qr,
  });
  const base = `http://127.0.0.1:${server.address().port}`;

  await checkAsync("wrong token yields 404 on landing page", async () => {
    const res = await fetch(`${base}/?t=wrong`);
    assert.strictEqual(res.status, 404);
  });

  await checkAsync("missing token yields 404", async () => {
    const res = await fetch(`${base}/`);
    assert.strictEqual(res.status, 404);
  });

  await checkAsync("qr.png with valid token but no QR yet yields 404", async () => {
    const res = await fetch(`${base}/qr.png?t=secret123`);
    assert.strictEqual(res.status, 404);
  });

  qr = "test-qr-payload";

  await checkAsync("qr.png with valid token and QR yields PNG", async () => {
    const res = await fetch(`${base}/qr.png?t=secret123`);
    assert.strictEqual(res.status, 200);
    assert.strictEqual(res.headers.get("content-type"), "image/png");
    const body = Buffer.from(await res.arrayBuffer());
    assert.ok(body.length > 0);
  });

  await checkAsync("landing page with valid token yields HTML with auto-refresh", async () => {
    const res = await fetch(`${base}/?t=secret123`);
    assert.strictEqual(res.status, 200);
    assert.strictEqual(res.headers.get("content-type"), "text/html");
    const body = await res.text();
    assert.ok(body.includes("auto-refresh"));
  });

  server.close();
  console.log(`${passed} checks passed`);
  console.log("ALL PASS");
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
