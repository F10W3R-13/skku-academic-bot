# Plan: whatsapp-web.js → Baileys 전환 (skku-whatsapp-bot)

목표: Chromium 상주(0.97GB)를 제거해 Railway 상시 RAM을 1.3GB → ~0.25GB로 낮춘다.
월 비용 ~$14 → Hobby $5 크레딧 내로. 봇의 대외 행동은 1도 달라지지 않는다.

판단 근거는 대화 실측: Railway 7일 메트릭(baseline 1.28GB flat, r=-0.021),
프로세스별 RSS(Chromium 75% / Python 127MB / Node 51MB).

## 0. 현행 행동 표면 (전부 보존 대상, id 순서 없음)

동작 유닛 — 하나라도 빠지면 리그레션:

| # | 동작 | 현재 구현 | 파일 |
|---|------|----------|------|
| S1 | 5개 그룹 수신 → `!ask`/@멘션 트리거 판정 | `GROUP_IDS` env (csv) | bot.js:147 |
| S2 | `!ask <q>` 프리픽스 파싱, `@멘션` 제거 파싱 | | trigger.js |
| S3 | 질문당 1회 쿨다운 10s (`COOLDOWN_MS`), chatId별 | | reply_guard.js |
| S4 | 동시 1질문 직렬화 (`busy`), "one at a time" 답변 | | bot.js:123 |
| S5 | 중복 이벤트 제거 (msg.id / 합성키, 500개 클리어) | | bot.js:85 |
| S6 | 타인 메시지 → 인용답장(msg.reply), 본인(fromMe) → sendMessage | | bot.js:87 |
| S7 | 본인 폰 `!ask` (message_create) 처리 | | bot.js:172 |
| S8 | 미등록 그룹 로깅 `[discover]` | | bot.js:153 |
| S9 | QR 터미널 + 웹페이지 (`/?t=TOKEN`, 12s 자동갱신) | | bot.js:58, qr_server.js |
| S10 | 답변 가드: 6000자 컷, 빈 답변 폴백 문구 | | reply_guard.js |
| S11 | API 60s 타임아웃, 실패 시 사과 문구 | | bot.js:47 |
| S12 | 크래시 복구: Railway 재시작 → LocalAuth 세션 생존 | | AUTH_DIR=/data |

비목표(전환과 무관, 손대지 않음): api.py 전체, 검색/BM25/캐시, QA 로깅,
회로 보호 로직 개선(쿨다운 회로 버그는 전환 후 별도 이슈로).

## 1. 사전 검증 (코드 한 줄 안 쓰고 실패 먼저 확인)

| 가설 | 검증법 | 통과 기준 |
|------|--------|----------|
| Baileys가 그룹 수신·발신·인용답장 가능 | P1 스파이크 로컬 실행 | 실제 그룹에서 3동작 확인 |
| 브라우저리스로 세션 유지 | P1 재시작 후 QR 없이 부팅 | creds 복원 로그 |
| RAM 목표 달성 | P1에서 process.memoryUsage | RSS < 150MB |
| 메시지 모양 매핑 정확성 | 유닛 테스트 | 어댑터 테스트 그린 |

Baileys 리스크 사실(무시 금지): 비공식 프로토콜 — WhatsApp 업데이트 시 깨질 수
있고 계정 밴 리스크 이론상 존재. 완화: 봇 번호는 전용번호 유지(개인번호 아님),
세션 죽으면 qr_server로 재인증(운영 절차 기존과 동일).
보안: Baileys 구버전(≤6.7.0)에 메시지 스푸핑 제로데이 이력(GHSA-qvv5-jq5g-4cgg)
— `baileys >= 7.0.0-rc12` 고정으로 완화. 업데이트 시 rc 최신 추적을 P5 운영 항목에 포함.

## 2. 설계

### 2.1 아키텍처: 어댑터 레이어

핵심 원칙: **트리거→쿨다운→API→가드 파이프라인은 전부 재사용**. 바뀌는 것은
"WhatsApp과 대화하는 법"뿐. 그 경계를 어댑터 한 파일로 몰아넣는다.

```
[whatsapp] ──► adapter (유일한 Baileys 의존 지점)
               │  translateIncoming(msg) → 표준 메시지 (기존 wwebjs 모양)
               │  sendMessage / reply / getGroups
[adapter] ───► processAsk (기존 로직 이식, wwebjs 타입 제거)
               ├─ trigger.js      (무수정 재사용)
               ├─ reply_guard.js  (무수정 재사용)
               └─ POST :8765/ask  (무수정 — api.py 건드리지 않음)
```

파일별 처분:

| 파일 | 처분 | 비고 |
|------|------|------|
| bot.js | **교체** | Baileys makeWASocket + 이벤트 바인딩. 파이프라인 로직은 이식 |
| adapter.js | **신규** | Baileys⇔표준 메시지 변환. Baileys import는 여기만 |
| trigger.js | 무수정 | 표준 메시지 객체만 받음 |
| reply_guard.js | 무수정 | 〃 |
| qr_server.js | 무수정 | getQr 콜백만 Baileys QR 문자열로 연결 |
| start.sh | 소폭 수정 | Chromium 관련 제거(SingletonLock 정리) |
| Dockerfile | 수정 | chromium 삭제, PLAY_credentials 플래그 제거 |
| api.py 이하 Python | **무수정** | |
| tests-js/* | 유지 + 어댑터 테스트 추가 | |

### 2.2 Baileys 기술 매핑 (P1에서 검증 후 확정)

패키지 팩트(2026-09-12 npm 실사 확인):
- 공식 패키지명은 **`baileys`**. `@whiskeysockets/baileys`는 deprecated(신규명 이관 공지) — 쓰지 않는다.
- 버전: `7.0.0-rc14` (latest tag, 활발한 유지보수, 주간 DL 60만). **6.7.0대 초반은
  메시지 스푸핑 제로데이(GHSA-qvv5-jq5g-4cgg) 이력이 있으므로 `>=7.0.0-rc12`로 고정.**
- README 공식: "Not running Chromium saves you like half a gig of ram" — 전환 근거와 정합.

| wwebjs (현재) | Baileys | 검증 포인트 |
|---|---|---|
| `client.initialize()` | `makeWASocket({ auth: state })` | creds 저장/복원 |
| `client.on('message')` | `messages.upsert` | 이벤트 배치 처리, fromMe 판정 |
| `msg.from` | `key.remoteJid` | `@g.us` 서픽스 동일 |
| `msg.fromMe` | `key.fromMe` | 〃 |
| `msg.id._serialized` | `key.id` | 중복제거 키 |
| `msg.body` | `message.conversation` 또는 `.extendedTextMessage.text` | 멘션 포함 시 후자 |
| `msg.mentionedIds` | `extendedTextMessage.contextInfo.mentionedJid` | @봇 트리거 |
| `msg.reply(text)` | `sock.sendMessage(jid, { text }, { quoted: msg.key })` | **S6 인용답장** |
| `client.sendMessage` | `sock.sendMessage(jid, { text })` | |
| `client.getChats()` | `sock.groupFetchAllParticipating()` | S1 그룹 로깅 |
| LocalAuth(dataPath) | `useMultiFileAuthState(dir)` | creds.json in /data |
| `qr` 이벤트 | `connection.update` `qr` 필드 | qr_server 연결 |
| disconnect 이벤트 | `connection.close` + `lastDisconnect.error` → `DisconnectReason` 매핑 | 재연결 정책 |

Baileys 도입부 (CJS — baileys는 ESM 기본 export라 interop 필요):

```js
const makeWASocket = require('baileys').default;
const { useMultiFileAuthState, fetchLatestBaileysVersion, DisconnectReason }
  = require('baileys');
```

P1에서 `require('baileys').default`가 Node20 CJS에서 실제로 resolve되는지 최우선 확인
(안 되면 ESM 변환 또는 dynamic import로 plan 갱신).

### 2.3 세션/저장소

- `useMultiFileAuthState('/data/baileys-auth')` — 볼륨 유지. 신규 폴더.
- **기존 wwebjs 세션(/data/session, /data/wwebjs_auth)은 이전 불가** — 프로토콜이
  다름. 1회 QR 재인증 필요. 전환 배포 후 qr_server 웹페이지로 스캔 (운영 절차 현행 유지).
- /data 정리: 전환 안정화 후 구 세션 폴더 수동 삭제 (볼륨 과금 미미하나 위생).

### 2.4 재연결 정책 (S12 대응)

connection.close statusCode → 의도적 로그아웃(401/`loggedOut`)만 creds 삭제+재QR,
나머지(515, 428, 440, 500 계열)는 백오프 재연결(1s→2s→4s→…→60s 캡). 무한 크래시
루프 방지를 위해 최근 5분 내 5회 실패 시 5분 대기. Railway 재시작은 언제든 safe.

## 3. 실행 계획 (페이즈 게이트)

### P0 — 준비 (반나절)
- [ ] `baileys@^7.0.0-rc12` 설치 (정확히 `baileys` 패키지. `@whiskeysockets/` 스코프 아님)
- [ ] CJS interop 확인: `node -e "console.log(typeof require('baileys').default)"` → function
- [ ] 로컬 .env에 세션 부트용 더미 QR_TOKEN 세팅
- [ ] 어댑터 인터페이스 시그니처 확정 (위 표) → 유닛 테스트 스캐폴딩

### P1 — 로컬 검증 스파이크 (반나절~하루) ⚠️ 유저 협력 필요
- [x] Baileys 실험 코드(spike_baileys.js) 로컬 실행
- [x] 페어링 코드로 실번호 링크 (QR 스캔 실패 → 페어링코드로 해결, 아래 교훈)
- [x] (d) 재시작 후 QR 없이 부팅 — creds.json 복원 확인
- [x] (e) RSS < 150MB — **실측 77–113MB** (부팅 110, 안정 77, GC 후 46)
- [ ] (a)(b)(c) 그룹 실측 — **테스트 그룹에서만** (아래 안전 절차)

안전 절차(필수): 실전 그룹은 절대 건드리지 않는다.
1. spike_baileys.js에 `SPIKE_ALLOW_JID` 필터 추가 — 이 JID 외에는 수신 자체를 무시(방어 1층)
2. 유저가 봇+본인만 넣은 **신규 테스트 그룹** 생성 (실제 학생 없음 — 방어 2층)
3. 테스트 그룹에서 !ask 실측: (a) 수신 (b) 인용답장 (c) 봇폰 fromMe !ask → sendMessage
4. Railway 봇이 같은 번호라 이중답변 여지가 있지만 테스트 그룹 안이면 무해

교훈(실측): `requestPairingCode`는 **딱 1회만** 호출한다. QR 로테이션(20초)마다
재호출하면 직전 코드가 전부 무효화되어 폰에서 '잘못된 코드'가 되고, 반복하면
세션이 loggedOut으로 붕괴한다. 코드 발급 후엔 로테이션과 무관하게 유효하므로
플래그로 1회만 요청하면 된다(스파이크에 수정 반영됨).

교훈(실측): 최초 연결 직후 히스토리 동기화가 `@lid` JID로 흘러온다. P2 어댑터는
수신 JID 정규화에서 lid/pn 구분을 고려해야 하고, 그룹 매칭(GROUP_IDS)은
`@g.us` JID 기준이라 이 영향은 그룹 경로에 없음 — 개인 DM 경로만 주의.

### P2 — 어댑터 + 파이프라인 이식 (1일)
- [ ] adapter.js 작성 (Baileys 의존 집중)
- [ ] bot.js 재작성 — processAsk 로직 이식하되 S5 중복제거는 어댑터 뒤로
- [ ] 어댑터 유닛 테스트 (Baileys 페이로드 → 표준 메시지 4케이스:
      conversation / extendedText+멘션 / fromMe / 비그룹)
- [ ] 기존 tests-js 3종 무수정 통과 확인

### P3 — 컨테이너 슬림화 (반나절)
- [ ] Dockerfile: chromium 제거, PUPPETEER_* env 제거, start.sh 정리
- [ ] 이미지 크기 확인 (chromium ~400MB 빠짐 — 빌드/디플로이도 빨라짐)
- [ ] Railway variables: 제거 대상 정리 (PUPPETEER 관련 없음 확인)

### P4 — 카나리 배포 (0.5일) ⚠️ 1회 QR 재인증 필요
- [ ] 배포 → qr_server 웹페이지로 QR 스캔 (유저 작업, 1분)
- [ ] 그룹에서 `!ask` 실측 (S1~S12 체크리스트)
- [ ] Railway 메트릭: baseline < 300MB 확인, 24h 관찰
- [ ] 이상 시 롤백: git revert + 재배포 (wwebjs 세션은 /data에 그대로 → 살아있음)

### P5 — 수습 (관찰)
- [ ] 3일간 메트릭/로그 모니터 → 문제 없으면 구 의존성 제거 커밋
- [ ] README 운영 절차 갱신 (QR 재인증, 세션 폴더 경로)
- [ ] (별도) S3 쿨다운 회로 버그 수정 이슈 등록

## 4. 롤백 전략

전환은 단일 트리(git)로 관리. 롤백 = `git revert <merge>` + 재배포.
wwebjs 세션 파일은 /data에 삭제하지 않고 남겨둠(P5 수습 기간) → 롤백 시 QR 없이 즉시 복원.

## 5. 성공 기준

| 항목 | 기준 |
|------|------|
| 기능 | S1~S12 체크리스트 전부 통과 (그룹 실측) |
| RAM | Railway baseline < 300MB (현재 1.28GB) |
| 비용 | Railway usage가 Hobby $5 크레딧 내 |
| 코드 | 기존 tests-js 3종 무수정 통과 + 어댑터 테스트 그린 |
| 롤백 가능성 | wwebjs 세션 보존으로 즉시 복원 가능 |
