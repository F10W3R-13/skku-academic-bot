const http = require("http");
const QRCode = require("qrcode");

function startQrServer({ port, getToken, getQr }) {
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, "http://localhost");
    const token = url.searchParams.get("t") || "";
    const expected = getToken();
    if (!expected || token !== expected) {
      res.writeHead(404, { "Content-Type": "text/plain" });
      return res.end("Not found");
    }
    if (url.pathname === "/qr.png") {
      const qr = getQr();
      if (!qr) { res.writeHead(404, {"Content-Type":"text/plain"}); return res.end("No QR yet"); }
      const buf = await QRCode.toBuffer(qr, { width: 480, margin: 2 });
      res.writeHead(200, { "Content-Type": "image/png", "Cache-Control": "no-store" });
      return res.end(buf);
    }
    // landing page: auto-refresh, big QR
    res.writeHead(200, { "Content-Type": "text/html" });
    res.end(`<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="12"><title>SKKU Bot QR</title></head><body style="text-align:center;font-family:sans-serif"><h2>WhatsApp QR — scans expire, page auto-refreshes</h2><p>WhatsApp &gt; Settings &gt; Linked Devices &gt; Link a Device</p><img src="/qr.png?t=${encodeURIComponent(token)}" alt="QR" width="480"></body></html>`);
  });
  return new Promise((resolve) => server.listen(port, "0.0.0.0", () => resolve(server)));
}

module.exports = { startQrServer };
