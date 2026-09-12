# SKKU Academic Regulations WhatsApp Bot — Design

Date: 2026-08-21
Status: Approved (pending implementation)

## Goal

Exchange students in a managed WhatsApp group ask academic questions in English.
A bot answers instantly using the local Korean markdown corpus of SKKU academic
regulations (`성균관대 학사제도/*.md`) as its knowledge base.

## Constraints & Decisions

| Decision | Choice | Reason |
|---|---|---|
| WhatsApp connection | whatsapp-web.js (unofficial) | Free, same-day setup, runs on user's own number via QR login |
| LLM | OpenAI `gpt-4o-mini` | User already has an OpenAI key |
| Retrieval | Embeddings RAG (`text-embedding-3-small`) | English questions must match Korean docs — cross-lingual semantic search |
| Hosting | User's Windows PC | No cost; PC must stay out of sleep mode |
| Scope | One specific WhatsApp group only | Safety: bot ignores DMs and other chats |
| Answer language | English (mirrors question language naturally) | Primary audience is exchange students |

## Architecture

Two processes on the user's PC, launched together by `start.bat`:

```
[WhatsApp group] ⇄ [bot.js — Node.js, whatsapp-web.js]
                          │  filters messages by group ID,
                          │  ignores own messages
                          ▼
                  [api.py — Python FastAPI, localhost:8765]
                          │  POST /ask {"question": "..."}
                          ▼
        [retrieve top-6 chunks via cosine similarity]
                          ▼
        [gpt-4o-mini generates grounded English answer]
                          ▼
        ["📚 Source: <document title>" appended]
```

Split rationale: whatsapp-web.js is Node-only; the RAG stack is cleanest in
Python. The two talk over plain HTTP on localhost.

## Components

### 1. `index_builder.py`

- Scans `*.md` in the project folder (skips `_`-prefixed files).
- Splits each file into chunks at `##`/`###` headings, keeping the heading
  path as context (e.g., `특별프로그램 > 현장실습 이수자격`). Chunks longer
  than ~1500 chars are split further at paragraph boundaries.
- Embeds each chunk with `text-embedding-3-small`.
- Writes `index.json`: `{source_hash, chunks: [{file, heading_path, text, embedding}]}`.
- `source_hash` = SHA-256 over all indexed file names + contents.
- CLI: `python index_builder.py [--force]`.

### 2. `api.py` (FastAPI)

- On startup: loads `index.json`; recomputes `source_hash` of the folder;
  if different, rebuilds the index automatically before serving.
- `POST /ask`:
  1. Embed the question.
  2. Cosine similarity → top 6 chunks.
  3. Prompt gpt-4o-mini: system prompt instructs it to answer in clear,
     friendly English using ONLY the provided context, cite which document
     each fact came from, say honestly when the context doesn't contain the
     answer, and suggest contacting the relevant SKKU office then.
  4. Returns `{answer, sources: [file titles]}`.
- `GET /health` for the bot to check readiness.

### 3. `bot.js` (Node.js, whatsapp-web.js)

- Uses `LocalAuth` so QR scan is needed once; session persists across restarts.
- Config in `config.json`: `{ "groupIds": ["..."], "apiUrl": "http://localhost:8765" }`.
- On startup, logs every chat's name + ID so the user can copy the target
  group ID into `config.json`.
- Message handling:
  - Only messages from IDs in `groupIds`.
  - Ignores its own messages and non-text messages.
  - Sends `⏳ Checking the regulations…` typing indicator while querying.
  - On success: replies with answer + `📚 Source: …` line.
  - On API unreachable/error: replies "Sorry, I'm having trouble right now.
    Please try again in a moment."
- Rate limit: max 1 concurrent query per chat; extra messages queue briefly
  or get a "one moment" nudge (keep simple: process sequentially).

### 4. `start.bat`

- Starts `uvicorn api:app --port 8765` in one window, `node bot.js` in another.
- Prints reminder: keep PC awake (Windows power settings).

## Data Update Flow

1. User drops new/updated `.md` files into the project folder
   (convert new PDFs with the existing `_pdf_to_md.py` workflow first).
2. Restart the bot (or run `python index_builder.py --force`).
   Hash mismatch triggers automatic re-embed (~a few seconds, negligible cost).

## Error Handling

| Failure | Behavior |
|---|---|
| OpenAI call fails/timeouts | Bot replies generic retry message |
| No relevant chunks found | LLM instructed to say it doesn't know + point to OIA/registrar contacts |
| WhatsApp disconnected | bot.js logs loudly; restart of start.bat re-authenticates via saved session |
| Malformed message (media etc.) | Silently ignored |

## Testing Plan

- Unit: chunker produces correct heading paths and sizes on real corpus.
- Integration: `api.py /ask` with sample English questions ("How many credits
  do I need to graduate?", "Can I take courses pass/fail?") returns grounded
  answers citing correct documents.
- E2E: private test WhatsApp group; send English question; verify reply,
  source line, and that DMs/other groups are ignored.
- Index refresh: modify one .md, restart, confirm updated answer.

## Out of Scope (YAGNI)

- Multi-group management UI, admin commands, conversation memory across
  messages, Korean-language document generation, cloud deployment.
