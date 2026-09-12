# SKKU Academic Regulations WhatsApp Bot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A WhatsApp bot that answers exchange students' English questions using the Korean markdown corpus of SKKU academic regulations via embeddings RAG.

**Architecture:** Two local processes — a Node.js whatsapp-web.js client that filters messages from one configured group, and a Python FastAPI service (`localhost:8765`) that retrieves top-6 document chunks by cosine similarity over OpenAI embeddings and generates a grounded English answer with gpt-4o-mini. All new code lives in `bot/`; the corpus stays in the parent folder.

**Tech Stack:** Node 18+, whatsapp-web.js, qrcode-terminal | Python 3.10+, FastAPI, uvicorn, openai (v1), numpy, python-dotenv, pytest

## Global Constraints

- Platform: Windows, PowerShell 5.1. Use `;` / `if ($?)` chaining, never `&&`.
- Corpus location: parent of `bot/` (i.e., `Path(__file__).resolve().parent.parent`). Index only `*.md` files whose names do NOT start with `_`.
- Models: `text-embedding-3-small` (embeddings), `gpt-4o-mini` (answers).
- API port: `8765`. Endpoint: `POST /ask` `{"question": str}` → `{"answer": str, "sources": [str]}`; `GET /health` → `{"status": "ok"}`.
- Answers in English; every successful reply ends with a `📚 Source:` line listing document titles.
- Chunk max length: 1500 chars. Top-k retrieval: 6.
- OpenAI key comes from env var `OPENAI_API_KEY`, loaded from `bot/.env` via python-dotenv.
- Git repo initialized inside `bot/` only (local, no remote). Commit after every task.
- Never print or commit the API key.

---

### Task 1: Scaffold + Markdown Chunker (TDD)

**Files:**
- Create: `bot/requirements.txt`
- Create: `bot/regulations/__init__.py` (empty)
- Create: `bot/regulations/chunker.py`
- Test: `bot/tests/test_chunker.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `chunk_markdown(raw_text: str, source_title: str) -> list[dict]` where each dict is `{"source": str, "heading_path": str, "text": str}`. Later tasks import `from regulations.chunker import chunk_markdown`.

- [ ] **Step 1: Verify toolchain**

Run: `python --version; node --version; pip --version`
Expected: Python ≥ 3.10, Node ≥ 18. If missing, stop and tell the user to install them.

- [ ] **Step 2: Scaffold + git init**

```powershell
New-Item -ItemType Directory -Force -Path bot\regulations, bot\tests | Out-Null
New-Item -ItemType File -Force -Path bot\regulations\__init__.py, bot\tests\__init__.py | Out-Null
```

(workdir: project root)

```powershell
git init
```

(workdir: `bot` — the repo lives inside `bot/` only, per Global Constraints. All later steps use workdir `bot` unless stated otherwise.)

Create `bot/requirements.txt`:

```
fastapi==0.115.*
uvicorn==0.32.*
openai>=1.50,<2
numpy>=1.26,<3
python-dotenv>=1.0
pytest>=8
httpx>=0.27
```

Run: `pip install -r requirements.txt`
Expected: all installed without error.

- [ ] **Step 3: Write failing tests**

Create `bot/tests/test_chunker.py`:

```python
from regulations.chunker import chunk_markdown

SAMPLE = """---
title: 학사일정표
source_pdf: x.pdf
---

# 학사일정표

## 1학기 일정

### 수강신청

- 1월 말 수강신청
- 정정은 2월 초

### 개강

- 3월 2일 개강한다. 아주 긴 문단이 이어진다. """ + ("가" * 1600) + """

## 등록

등록금은 2월에 납부한다.
"""


def test_frontmatter_is_skipped():
    chunks = chunk_markdown(SAMPLE, "학사일정표")
    assert all("source_pdf" not in c["text"] for c in chunks)


def test_heading_paths_are_joined():
    chunks = chunk_markdown(SAMPLE, "학사일정표")
    paths = [c["heading_path"] for c in chunks]
    assert "학사일정표 > 1학기 일정 > 수강신청" in paths
    assert "학사일정표 > 등록" in paths


def test_every_chunk_has_source_and_text():
    chunks = chunk_markdown(SAMPLE, "학사일정표")
    assert len(chunks) >= 3
    for c in chunks:
        assert c["source"] == "학사일정표"
        assert c["text"].strip()


def test_long_sections_are_split():
    chunks = chunk_markdown(SAMPLE, "학사일정표")
    assert all(len(c["text"]) <= 1500 for c in chunks)


def test_empty_input():
    assert chunk_markdown("", "x") == []
```

- [ ] **Step 4: Run tests, verify failure**

Run (workdir `bot`): `python -m pytest tests/test_chunker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'regulations'`.

- [ ] **Step 5: Implement chunker**

Create `bot/regulations/chunker.py`:

```python
import re

MAX_CHARS = 1500


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            return text[end + 3:].lstrip("\n")
    return text


def _split_long(text: str, limit: int = MAX_CHARS) -> list[str]:
    parts: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n", text):
        while len(para) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(para[:limit])
            para = para[limit:]
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) > limit and current:
            parts.append(current)
            current = para
        else:
            current = candidate
    if current.strip():
        parts.append(current)
    return parts


def chunk_markdown(raw_text: str, source_title: str) -> list[dict]:
    body = _strip_frontmatter(raw_text)
    if not body.strip():
        return []
    lines = body.splitlines()
    chunks: list[dict] = []
    h2, h3 = "", ""
    buf: list[str] = []

    def flush():
        text = "\n".join(buf).strip()
        buf.clear()
        if not text:
            return
        path = " > ".join(p for p in (source_title, h2, h3) if p)
        for piece in _split_long(text):
            chunks.append({"source": source_title, "heading_path": path, "text": piece})

    for line in lines:
        m2 = re.match(r"^##\s+(?!#)(.+)", line)
        m3 = re.match(r"^###\s+(.+)", line)
        if m2:
            flush()
            h2, h3 = m2.group(1).strip(), ""
        elif m3:
            flush()
            h3 = m3.group(1).strip()
        else:
            buf.append(line)
    flush()
    return chunks
```

- [ ] **Step 6: Run tests, verify pass**

Run (workdir `bot`): `python -m pytest tests/test_chunker.py -v`
Expected: 5 passed.

- [ ] **Step 7: Smoke-test on the real corpus**

```powershell
python -c "from pathlib import Path; from regulations.chunker import chunk_markdown; files=[p for p in Path('..').glob('*.md') if not p.name.startswith('_')]; total=sum(len(chunk_markdown(p.read_text(encoding='utf-8'), p.stem)) for p in files); print(f'{len(files)} files, {total} chunks')"
```

Expected: `13 files, N chunks` with N roughly between 100 and 600.

- [ ] **Step 8: Commit**

```powershell
git add -A; if ($?) { git commit -m "feat: markdown chunker with heading-path awareness" }
```

(workdir `bot`)

---

### Task 2: Index Builder (hash + embeddings)

**Files:**
- Create: `bot/regulations/index_builder.py`
- Create: `bot/regulations/openai_client.py`
- Test: `bot/tests/test_index_builder.py`

**Interfaces:**
- Consumes: `chunk_markdown` from Task 1.
- Produces:
  - `compute_source_hash(md_files: list[Path]) -> str` (SHA-256 hex of sorted `"name\ncontent"` pairs)
  - `list_corpus_files(folder: Path) -> list[Path]` (parent-dir `*.md`, skipping `_` prefix)
  - `build_index(folder: Path, index_path: Path, force: bool = False) -> bool` — rebuilds if forced or hash changed; writes `index.json`; returns True if rebuilt.
  - `openai_client.get_client()` → shared `OpenAI` instance; `openai_client.embed_texts(texts: list[str]) -> list[list[float]]`.
  - `index.json` schema: `{"source_hash": str, "chunks": [{"source": str, "heading_path": str, "text": str, "embedding": [float]}]}`

- [ ] **Step 1: Write failing tests**

Create `bot/tests/test_index_builder.py`:

```python
import hashlib
import json
from pathlib import Path

from regulations.index_builder import compute_source_hash, list_corpus_files


def test_list_corpus_files_skips_underscore(tmp_path: Path):
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "_skip.md").write_text("s", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    names = [p.name for p in list_corpus_files(tmp_path)]
    assert names == ["a.md"]


def test_source_hash_changes_when_content_changes(tmp_path: Path):
    f = tmp_path / "a.md"
    f.write_text("v1", encoding="utf-8")
    h1 = compute_source_hash([f])
    f.write_text("v2", encoding="utf-8")
    h2 = compute_source_hash([f])
    assert h1 != h2
    assert h1 == hashlib.sha256(b"a.md\nv1").hexdigest()


def test_source_hash_order_independent(tmp_path: Path):
    a = tmp_path / "a.md"; b = tmp_path / "b.md"
    a.write_text("A", encoding="utf-8"); b.write_text("B", encoding="utf-8")
    assert compute_source_hash([a, b]) == compute_source_hash([b, a])
```

- [ ] **Step 2: Run tests, verify failure**

Run (workdir `bot`): `python -m pytest tests/test_index_builder.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `bot/regulations/openai_client.py`:

```python
import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

load_dotenv()


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in bot/.env")
    return OpenAI()


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = get_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend(d.embedding for d in resp.data)
    return out
```

Create `bot/regulations/index_builder.py`:

```python
import hashlib
import json
from pathlib import Path

from regulations.chunker import chunk_markdown
from regulations.openai_client import embed_texts


def list_corpus_files(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.glob("*.md")
        if not p.name.startswith("_") and not p.name.startswith("~$")
    )


def compute_source_hash(md_files: list[Path]) -> str:
    sha = hashlib.sha256()
    for p in sorted(md_files, key=lambda x: x.name):
        sha.update(p.name.encode("utf-8"))
        sha.update(b"\n")
        sha.update(p.read_bytes())
    return sha.hexdigest()


def build_index(folder: Path, index_path: Path, force: bool = False) -> bool:
    files = list_corpus_files(folder)
    if not files:
        raise FileNotFoundError(f"No .md corpus files found in {folder}")
    current_hash = compute_source_hash(files)
    if not force and index_path.exists():
        old = json.loads(index_path.read_text(encoding="utf-8"))
        if old.get("source_hash") == current_hash:
            return False

    chunks = []
    for p in files:
        raw = p.read_text(encoding="utf-8")
        chunks.extend(chunk_markdown(raw, p.stem))

    embeddings = embed_texts([c["text"] for c in chunks])
    for c, emb in zip(chunks, embeddings):
        c["embedding"] = emb

    index_path.write_text(
        json.dumps({"source_hash": current_hash, "chunks": chunks}, ensure_ascii=False),
        encoding="utf-8",
    )
    return True
```

- [ ] **Step 4: Run tests, verify pass**

Run (workdir `bot`): `python -m pytest tests/test_index_builder.py -v`
Expected: 3 passed.

- [ ] **Step 5: Build the real index (needs OPENAI_API_KEY)**

Ask the user for their key, then create `bot/.env` containing exactly one line `OPENAI_API_KEY=sk-...` (never commit `.env`; verify `.gitignore` covers it).

Create `bot/.gitignore`:

```
.env
index.json
.wwebjs_auth/
.wwebjs_cache/
__pycache__/
node_modules/
.pytest_cache/
```

Run (workdir `bot`):

```powershell
python -c "from pathlib import Path; from regulations.index_builder import build_index; rebuilt=build_index(Path('..').resolve(), Path('index.json')); print('rebuilt:', rebuilt)"
```

Expected: `rebuilt: True`, `index.json` created (a few hundred KB–few MB). Run the same command again → `rebuilt: False` (hash unchanged).

- [ ] **Step 6: Commit**

```powershell
git add -A; if ($?) { git commit -m "feat: embedding index builder with content-hash change detection" }
```

(workdir `bot`)

---

### Task 3: FastAPI Service (/ask)

**Files:**
- Create: `bot/api.py`
- Test: `bot/tests/test_api.py`

**Interfaces:**
- Consumes: `build_index`, `list_corpus_files`, `compute_source_hash` (Task 2); `embed_texts`, `CHAT_MODEL` (Task 2).
- Produces: HTTP `GET /health` → `{"status":"ok"}`; `POST /ask` body `{"question": str}` → `200 {"answer": str, "sources": [str]}`. `bot.js` (Task 4) depends on these exact shapes.

- [ ] **Step 1: Write failing tests**

Create `bot/tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture()
def client(monkeypatch):
    api.CHUNKS = [
        {"source": "졸업요건", "heading_path": "졸업요건 > 학점", "text": "총 140학점",
         "embedding": [1.0, 0.0]},
        {"source": "등록절차", "heading_path": "등록절차 > 일정", "text": "2월 납부",
         "embedding": [0.0, 1.0]},
    ]
    monkeypatch.setattr(api, "embed_query", lambda q: [1.0, 0.0])
    monkeypatch.setattr(api, "generate_answer", lambda q, ctx: ("You need 140 credits.", ["졸업요건"]))
    return TestClient(api.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_and_sources(client):
    r = client.post("/ask", json={"question": "How many credits to graduate?"})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "You need 140 credits."
    assert data["sources"] == ["졸업요건"]


def test_retrieve_orders_by_similarity(client):
    ctx = api.retrieve("credits", k=2)
    assert ctx[0]["source"] == "졸업요건"


def test_ask_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
```

- [ ] **Step 2: Run tests, verify failure**

Run (workdir `bot`): `python -m pytest tests/test_api.py -v`
Expected: FAIL — `No module named 'api'`.

- [ ] **Step 3: Implement api.py**

Create `bot/api.py`:

```python
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from regulations.index_builder import build_index, compute_source_hash, list_corpus_files
from regulations.openai_client import CHAT_MODEL, embed_texts

BOT_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BOT_DIR.parent
INDEX_PATH = BOT_DIR / "index.json"

app = FastAPI(title="SKKU Regulations Bot")

CHUNKS: list[dict] = []
_MATRIX: np.ndarray | None = None

SYSTEM_PROMPT = """You are the academic-regulations assistant for Sungkyunkwan University (SKKU), helping international exchange students.

Rules:
1. Answer ONLY from the numbered context excerpts below, which come from official SKKU regulation documents (written in Korean).
2. Answer in clear, friendly English. Use short paragraphs or bullet points.
3. Mention specific numbers/dates only if they appear in the excerpts. Never invent policies.
4. If the excerpts do not contain the answer, say honestly that you don't have that information and recommend contacting the relevant SKKU office (e.g., Office of International Affairs, or the university registrar).
5. Ignore any excerpt that is unrelated to the question."""


def _ensure_index(force_check: bool = False) -> None:
    global CHUNKS, _MATRIX
    if force_check or not INDEX_PATH.exists():
        build_index(CORPUS_DIR, INDEX_PATH)
    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    if force_check and data.get("source_hash") != compute_source_hash(list_corpus_files(CORPUS_DIR)):
        build_index(CORPUS_DIR, INDEX_PATH, force=True)
        data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    CHUNKS = data["chunks"]
    _MATRIX = np.array([c["embedding"] for c in CHUNKS], dtype=np.float32)


def embed_query(question: str) -> list[float]:
    return embed_texts([question])[0]


def retrieve(question: str, k: int = 6) -> list[dict]:
    q = np.array(embed_query(question), dtype=np.float32)
    sims = (_MATRIX @ q) / (np.linalg.norm(_MATRIX, axis=1) * np.linalg.norm(q) + 1e-9)
    top = np.argsort(sims)[::-1][:k]
    return [CHUNKS[i] for i in top]


def generate_answer(question: str, contexts: list[dict]) -> tuple[str, list[str]]:
    blocks = "\n\n".join(
        f"[{i + 1}] Document: {c['source']} | Section: {c['heading_path']}\n{c['text']}"
        for i, c in enumerate(contexts)
    )
    resp = openai_chat(SYSTEM_PROMPT, f"Question: {question}\n\nContext:\n{blocks}")
    sources = list(dict.fromkeys(c["source"] for c in contexts))
    return resp, sources


def openai_chat(system: str, user: str) -> str:
    from regulations.openai_client import get_client
    resp = get_client().chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        max_tokens=700,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content.strip()


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        contexts = retrieve(req.question)
        if not contexts:
            raise ValueError("empty index")
        answer, sources = generate_answer(req.question, contexts)
        return {"answer": answer, "sources": sources}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.on_event("startup")
def startup():
    # Runs under uvicorn only — importing api.py in tests must NOT touch the network.
    _ensure_index(force_check=True)
```

- [ ] **Step 4: Run tests, verify pass**

Run (workdir `bot`): `python -m pytest tests/ -v`
Expected: all passed (chunker + index + api).

- [ ] **Step 5: Live smoke test**

Run (workdir `bot`): `python -m uvicorn api:app --port 8765`
In a second shell:

```powershell
Invoke-RestMethod -Uri http://localhost:8765/health; Invoke-RestMethod -Uri http://localhost:8765/ask -Method Post -ContentType "application/json" -Body '{"question":"How many credits do I need to graduate?"}' | ConvertTo-Json -Depth 3
```

Expected: `{"status":"ok"}` then an English answer mentioning credit requirements with `sources` containing `졸업요건`. Stop the server afterwards (Ctrl+C).

- [ ] **Step 6: Commit**

```powershell
git add -A; if ($?) { git commit -m "feat: FastAPI RAG endpoint with grounded English answers" }
```

(workdir `bot`)

---

### Task 4: WhatsApp Client (bot.js)

**Files:**
- Create: `bot/package.json`
- Create: `bot/config.json`
- Create: `bot/bot.js`

**Interfaces:**
- Consumes: `POST {apiUrl}/ask {"question"}` → `{"answer","sources"}` (Task 3); `config.json` `{"groupIds":[str],"apiUrl":str}`.
- Produces: working WhatsApp listener. No other file imports it.

- [ ] **Step 1: Install dependencies**

Create `bot/package.json`:

```json
{
  "name": "skku-whatsapp-bot",
  "version": "1.0.0",
  "private": true,
  "main": "bot.js",
  "scripts": { "start": "node bot.js" },
  "dependencies": {
    "whatsapp-web.js": "^1.26.0",
    "qrcode-terminal": "^0.12.0"
  }
}
```

Create `bot/config.json`:

```json
{ "groupIds": [], "apiUrl": "http://localhost:8765" }
```

Run (workdir `bot`): `npm install`
Expected: installs, Puppeteer Chromium downloaded. First run can take several minutes.

- [ ] **Step 2: Implement bot.js**

Create `bot/bot.js`:

```js
const fs = require("fs");
const { Client, LocalAuth } = require("whatsapp-web.js");
const qrcode = require("qrcode-terminal");

const config = JSON.parse(fs.readFileSync("./config.json", "utf8"));
if (!Array.isArray(config.groupIds) || config.groupIds.length === 0) {
  console.log("[setup] config.json groupIds is empty.");
  console.log("[setup] After startup, copy your group's ID below into config.json and restart.");
}

const client = new Client({
  authStrategy: new LocalAuth(),
  puppeteer: { args: ["--no-sandbox"] },
});

let busy = false;

async function askApi(question) {
  const res = await fetch(`${config.apiUrl}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error(`API responded ${res.status}`);
  return res.json();
}

client.on("qr", (qr) => {
  console.log("[auth] Scan this QR in WhatsApp: Settings > Linked Devices > Link a Device");
  qrcode.generate(qr, { small: true });
});

client.on("ready", async () => {
  console.log("[ready] WhatsApp connected.");
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
    if (!config.groupIds.includes(msg.from)) return;
    const body = (msg.body || "").trim();
    if (!body) return;

    if (busy) {
      await msg.reply("One moment please — I answer one question at a time. 🙏");
      return;
    }
    busy = true;
    console.log(`[q] ${body}`);
    let data;
    try {
      data = await askApi(body);
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
```

- [ ] **Step 3: Syntax check**

Run (workdir `bot`): `node --check bot.js`
Expected: no output (success).

- [ ] **Step 4: Manual E2E (requires user participation)**

1. Start API: `python -m uvicorn api:app --port 8765` (workdir `bot`).
2. Start bot: `node bot.js` (workdir `bot`).
3. Scan QR with the user's phone.
4. Copy the target group's ID from the `[group]` log lines into `config.json` `groupIds`, restart `node bot.js`.
5. From another account in that group, send: `How many credits do I need to graduate?`
6. Expected: bot replies an English answer ending with `📚 Source: 졸업요건` (or similar).
7. Send a DM to the bot's number → expected: no reply.
8. Stop the API server, send another group message → expected: "Sorry, I'm having trouble right now…" reply.

- [ ] **Step 5: Commit**

```powershell
git add -A; if ($?) { git commit -m "feat: whatsapp-web.js group bot wired to RAG API" }
```

(workdir `bot`)

---

### Task 5: Launcher + README

**Files:**
- Create: `bot/start.bat`
- Create: `bot/README.md`

**Interfaces:**
- Consumes: everything above. Produces: one-double-click launcher + setup docs.

- [ ] **Step 1: Create start.bat**

```bat
@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [1/3] Starting regulations API...
start "SKKU Bot - API" cmd /k "python -m uvicorn api:app --port 8765"

echo [2/3] Waiting for API...
timeout /t 6 /nobreak >nul

echo [3/3] Starting WhatsApp bot...
start "SKKU Bot - WhatsApp" cmd /k "node bot.js"

echo.
echo Both windows are running. Keep this PC awake
echo (Settings > System > Power > Sleep: Never).
echo Close both windows to stop the bot.
pause
```

- [ ] **Step 2: Verify launcher works**

Double-click `bot/start.bat` (or run it from PowerShell). Expected: two windows open; API window shows uvicorn startup; WhatsApp window shows QR (first time) or `[ready]` (after session saved). Close both windows.

- [ ] **Step 3: Write README.md**

Create `bot/README.md`:

```markdown
# SKKU 학사제도 WhatsApp 봇

교환학생 그룹톡에서 영어 질문에 학사제도 기준으로 자동 답변하는 봇.

## 최초 설정 (1회)

1. 요구사항: Python 3.10+, Node 18+
2. `pip install -r requirements.txt`
3. `npm install`
4. `bot/.env` 파일 만들고 한 줄: `OPENAI_API_KEY=sk-...`
5. 인덱스 생성: `python -c "from pathlib import Path; from regulations.index_builder import build_index; build_index(Path('..').resolve(), Path('index.json'), force=True)"`
6. `start.bat` 실행 → QR 스캔 → 콘솔의 `[group] 이름 | ID` 목록에서
   그룹 ID를 복사해 `config.json`의 `groupIds`에 넣고 WhatsApp 창 재시작.

## 매일 사용

- `start.bat` 더블클릭 → 두 창이 뜨면 완료. QR은 다시 묻지 않음.
- PC 절전 모드 해제 필수: 설정 > 시스템 > 전원 > 절전 "안 함".

## 데이터 갱신

- 부모 폴더에 새 `.md` 추가/수정 (새 PDF는 `_pdf_to_md.py`로 먼저 변환)
- 봇 재시작하면 변경 감지 시 자동 재임베딩. 강제 재빌드는 위 5번 명령에 `force=True`.

## 문제 해결

| 증상 | 조치 |
|---|---|
| 봇이 무응답 | API 창과 WhatsApp 창이 모두 켜져 있는지 확인 |
| QR 다시 요구 | `.wwebjs_auth` 폴더 삭제 후 재스캔 |
| "having trouble" 답장 | API 창 에러 확인 (대부분 OPENAI_API_KEY 문제) |
```

- [ ] **Step 4: Full test suite + final commit**

Run (workdir `bot`): `python -m pytest tests/ -v; node --check bot.js`
Expected: all tests pass, syntax OK.

```powershell
git add -A; if ($?) { git commit -m "docs: launcher and setup guide" }
```

(workdir `bot`)
