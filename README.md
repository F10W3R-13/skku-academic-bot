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
- 봇 호출 방법: 메시지를 `!ask ` 로 시작하거나, 봇 계정(@내계정)을 멘션하세요. 그 외 일반 잡담에는 반응하지 않습니다.
- `!ask`만 치고 아무것도 안 쓰면 사용법 안내를 답장합니다.

## 데이터 갱신

- `bot/corpus/` 폴더에 새 `.md` 추가/수정 (새 PDF는 `_pdf_to_md.py`로 먼저 변환). 개인 성적·학적 파일은 절대 넣지 마세요.
- 인덱서는 `corpus/*.md` 만 훑습니다(**하위 폴더 미탐색**). 자료는 평평하게 넣고, 출처 구분은 파일명 접두사로 하세요 (예: `학교생활_*.md`).
- 봇 재시작하면 변경 감지 시 자동 재임베딩. 강제 재빌드는 위 5번 명령에 `force=True`.

## .env 설정값

| 키 | 기본값 | 설명 |
|---|---|---|
| `OPENAI_API_KEY` | (필수) | |
| `CHAT_MODEL` | `gpt-5.6-luna` | 답변 생성 모델. 되돌리려면 `CHAT_MODEL=gpt-4o-mini` |
| `EMBED_MODEL` | `text-embedding-3-small` | 검색용 임베딩. **바꾸면 인덱스가 자동 전체 재빌드됨** |
| `MAX_ANSWER_TOKENS` | `700` | 답변 길이. 400=요점만, 1200=절차를 단계별로 |
| `MAX_QUERY_TOKENS` | `80` | 내부 질의 확장용(건드릴 일 거의 없음) |
| `MAX_CONTEXT_CHARS` | `12000` | 모델에 넣는 발췌문 총량 상한. 한 문서가 여러 조각으로 갈려 사실이 흩어지는 걸 보완하되, 큰 문서가 컨텍스트를 다 먹지 않게 막는다 |
| `CORPUS_DIR` | `bot/corpus` | |
| `INDEX_PATH` | `bot/index.json` | |
| `QA_LOG_DIR` | `bot/logs` | 질문·답변 QA 로그 폴더 |

## QA 로그 (답변 품질 조사)

- 모든 `!ask` 질문과 답변이 `logs/qa_YYYYMMDD.jsonl` 에 하루 파일 하나로 쌓인다.
  기록 필드: 시각, 질문, 답변, 근거 문서(`sources`), 근거 발췌 원문(`contexts`), 검색 최고점수(`retrieval.top_score`), 소요시간, 성공/실패. `contexts` 덕분에 나중에 API 재호출 없이 로그만으로 답변 충실성 심판이 가능하다.
- **발신자·그룹 ID는 기록하지 않는다** (API 계층에 그 정보가 없다). 로그 폴더는 커밋 제외(`.gitignore`).
- 답변이 허술했던 질문 뽑기: `python _analyze_qa.py` (기본 `logs/`, `--days 7` 로 최근 일주일만).
  검색 최고점수가 0.45 미만이면 코퍼스에 없는 주제일 가능성이 큼 → 자료 보강(크롤링) 1순위.
- 로그 폴더 변경: `.env`에 `QA_LOG_DIR=...` (Railway 등에서 볼륨 경로로 지정).

## 검색이 이상할 때

- `python _check_retrieval.py "질문"` 실행 → 한국어로 어떻게 번역돼 검색됐는지 + 상위 10개 청크와 점수를 보여줍니다.
- 관련 문서가 아예 안 잡히면 자료 부족, 엉뚱한 문서가 높은 점수면 검색 문제입니다.
- 코퍼스가 한국어인데 질문은 영어라 언어가 어긋납니다. `api.py`의 `expand_queries()`가 질문을 한국어 검색어로 한 번 더 번역해 두 결과를 합칩니다.

## Railway 배포 (24시간 클라우드 운영)

1. 이 `bot/` 폴더를 GitHub **프라이빗** 저장소로 푸시
2. railway.app에서 New Project → Deploy from GitHub repo 선택
3. Variables에 등록:
   - `OPENAI_API_KEY` (필수)
   - `AUTH_DIR=/data` (세션 저장용)
   - `INDEX_PATH=/data/index.json`
   - `CORPUS_DIR=/app/corpus`
   - `GROUP_IDS`는 첫 배포 후 로그의 `[group] 이름 | ID`를 보고 추가 후 재배포
4. Settings → Volumes → `/data` 볼륨 연결 (재배포해도 QR 재스캔 안 함)
5. 첫 배포 로그에서 QR 스캔 1회
6. 주의: 로컬 PC와 동시 실행 금지 (중복 답변 발생)

### 원격 QR 스캔

- Variables에 `QR_TOKEN`에 아무 긴 문자열 추가 → 재배포
- Railway의 Public Networking에서 도메인 생성 후
  `https://<도메인>/?t=<QR_TOKEN>` 을 브라우저로 열면 큰 QR 이미지가 뜸 (12초 자동 갱신)
- 토큰 없는 접속은 404. QR은 연결되기 전까지만 유효.

### 그룹 추가하기

1. 봇 계정(내 번호)을 새 그룹에 초대 — 내가 이미 멤버인 방은 자동으로 보임
2. 그 그룹에서 아무나 메시지 하나 전송 → 로그에 `[discover] group not in GROUP_IDS: <ID>` 한 줄 기록
3. Railway Variables의 `GROUP_IDS`에 쉼표로 이어서 추가 → 재배포
   예: `120363428912675030@g.us,987654321098765432@g.us`
4. 제거할 때도 GROUP_IDS에서 해당 ID만 빼고 재배포

## 문제 해결

| 증상 | 조치 |
|---|---|
| 봇이 무응답 | API 창과 WhatsApp 창이 모두 켜져 있는지 확인 |
| QR 다시 요구 | `.wwebjs_auth` 폴더 삭제 후 재스캔 |
| "having trouble" 답장 | API 창 에러 확인 (대부분 OPENAI_API_KEY 문제) |
