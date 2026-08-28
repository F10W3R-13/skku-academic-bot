"""코퍼스 린터 — 인덱싱 전에 돌려 데이터 품질 문제를 미리 잡는다.

봇 실행·API 호출 없음. 사용법 (bot/ 디렉터리에서):
    py -3.13 _lint_corpus.py            # corpus/ 대상
    py -3.13 _lint_corpus.py some_dir   # 다른 폴더 대상

종료 코드: ERROR 있으면 1 (인덱싱 중단 권장), WARN만 있으면 0.

검사 항목:
  [ERROR] 제작 잔여물 — 📎, TODO, FIXME. 스크래핑/변환 과정의 메타 노트가
          본문에 남으면 그대로 임베딩되어 검색을 오염시킨다(실측 사고).
  [ERROR] 본문 없는 문서 — frontmatter 제외 본문 200자 미만. 동적 위젯
          페이지를 md로 굳힌 껍데기 등(실측: SKKU 캘린더).
  [WARN]  제목 세그먼트 충돌 — 두 문서 제목이 같은 핵심 단어를 공유.
          어휘(BM25) 가점이 동시에 점수를 줘 검색 경합이 난다(실측: 증명서발급
          이원화 → 요금 오답). 병합하거나 제목을 달라지게 할 것.
  [WARN]  초소 문서(1,000자 미만) — 정상일 수 있으나 확인 권장.
  [WARN]  과대 청크(2,000자 초과) — 절 구조가 없는 통짜 문서 신호.
"""
import re
import sys
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BOT_DIR))

from regulations.chunker import chunk_markdown  # noqa: E402

STOP_SEGMENTS = {"학교생활", "서울생활", "등", "and", "the", "of"}


def title_segments(name: str) -> set[str]:
    return {
        p
        for p in re.split(r"[\s_()\-/,|.·:]+", name)
        if len(p) >= 2 and p.casefold() not in STOP_SEGMENTS
    }


def body_text(raw: str) -> str:
    """frontmatter(--- ... ---)를 제외한 본문."""
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            return raw[end + 4:]
    return raw


def main() -> int:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else BOT_DIR / "corpus"
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    errors: list[str] = []
    warns: list[str] = []
    files = sorted(p for p in folder.glob("*.md") if not p.name.startswith(("_", "~$")))
    if not files:
        print(f"corpus 없음: {folder}")
        return 1

    segs: dict[str, set[str]] = {}
    for p in files:
        raw = p.read_text(encoding="utf-8")
        body = body_text(raw)

        for pat, desc in [("📎", "📎 메타노트"), ("TODO", "TODO"), ("FIXME", "FIXME")]:
            if pat in raw:
                errors.append(f"{p.name}: 제작 잔여물 {desc}")
        if len(body.strip()) < 200:
            errors.append(f"{p.name}: 본문 {len(body.strip())}자 — 빈 껍데기 문서")
        elif len(body.strip()) < 1000:
            warns.append(f"{p.name}: {len(body.strip())}자 (짧음)")

        for c in chunk_markdown(raw, p.stem):
            if len(c["text"]) > 2000:
                warns.append(f"{p.name}: 청크 {len(c['text']):,}자 초과 ({c['heading_path'][:30]})")
                break
        segs[p.stem] = title_segments(p.stem)

    names = sorted(segs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = segs[a] & segs[b]
            if common:
                warns.append(f"제목 충돌 {sorted(common)}: {a} <-> {b}")

    print(f"검사 대상 {len(files)}개 문서")
    if errors:
        print(f"\n[ERROR] {len(errors)}개 — 인덱싱 전 수정 권장")
        for e in errors:
            print(f"  - {e}")
    if warns:
        print(f"\n[WARN] {len(warns)}개")
        for w in warns:
            print(f"  - {w}")
    if not errors and not warns:
        print("문제 없음")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
