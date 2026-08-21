#!/bin/bash
set -e
echo "[start] launching RAG API..."
python -m uvicorn api:app --host 0.0.0.0 --port 8765 &
echo "[start] waiting for API health..."
ok=0
for i in $(seq 1 60); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=2)" 2>/dev/null; then
    echo "[start] API healthy"; ok=1; break
  fi
  sleep 1
done
[ "$ok" = "1" ] || { echo "[start] API failed to become healthy"; exit 1; }

AUTH_DIR="${AUTH_DIR:-.}"
rm -f "$AUTH_DIR"/.wwebjs_auth/session/Singleton* 2>/dev/null \
  && echo "[start] cleared stale Chromium locks (if any)" || true

exec node bot.js
