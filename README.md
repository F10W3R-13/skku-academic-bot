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

- 부모 폴더에 새 `.md` 추가/수정 (새 PDF는 `_pdf_to_md.py`로 먼저 변환)
- 봇 재시작하면 변경 감지 시 자동 재임베딩. 강제 재빌드는 위 5번 명령에 `force=True`.

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

## 문제 해결

| 증상 | 조치 |
|---|---|
| 봇이 무응답 | API 창과 WhatsApp 창이 모두 켜져 있는지 확인 |
| QR 다시 요구 | `.wwebjs_auth` 폴더 삭제 후 재스캔 |
| "having trouble" 답장 | API 창 에러 확인 (대부분 OPENAI_API_KEY 문제) |
