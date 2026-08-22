"""questions.jsonl 스크립트·근거 검증 (봇 실행 없음, API 호출 없음).

확인 항목:
    1. 모든 줄이 JSON으로 파싱되고 필수 필드가 있는지
    2. category 허용값(single/multi/absent/robust/adversarial), id 중복 없음
    3. expected_docs 스템이 corpus의 실제 파일과 일치하는지
    4. 모든 fact의 context_any가 지정 문서 원문에 실제 존재하는지
       — needle을 지어냈으면(환각) 여기서 걸린다
    5. answer_any가 비어 있지 않은지

사용법:
    python eval/validate_gold.py [questions.jsonl 경로]
종료 코드: 성공 0, 오류 있으면 1
"""
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BOT_DIR = EVAL_DIR.parent
CORPUS = BOT_DIR / "corpus"
ALLOWED_CATEGORIES = {"single", "multi", "absent", "robust", "adversarial"}


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else EVAL_DIR / "questions.jsonl"
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    errors: list[str] = []
    seen_ids: set[str] = set()
    corpus_texts: dict[str, str] = {}

    lines = target.read_text(encoding="utf-8").splitlines()
    for no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{no}줄: JSON 파싱 실패 - {exc}")
            continue

        qid = item.get("id", f"{no}줄( id 없음 )")
        for field in ["id", "category", "question", "expected_docs", "facts"]:
            if field not in item:
                errors.append(f"{qid}: 필수 필드 누락 - {field}")
        if item.get("category") not in ALLOWED_CATEGORIES:
            errors.append(f"{qid}: 알 수 없는 category - {item.get('category')}")
        if item["id"] in seen_ids:
            errors.append(f"{qid}: id 중복")
        seen_ids.add(item["id"])

        for doc in item.get("expected_docs", []):
            if doc not in corpus_texts:
                path = CORPUS / f"{doc}.md"
                if not path.exists():
                    errors.append(f"{qid}: 코퍼스에 문서 없음 - {doc}")
                    continue
                corpus_texts[doc] = path.read_text(encoding="utf-8")

        for f in item.get("facts", []):
            if not f.get("answer_any"):
                errors.append(f"{qid}: answer_any 비어 있음 - {f.get('label')}")
            for needle in f.get("context_any", []):
                if not any(
                    needle in corpus_texts.get(doc, "") for doc in item["expected_docs"]
                ):
                    errors.append(
                        f"{qid}: context_any가 원문에 없음 - '{needle}' "
                        f"({f.get('label')}) in {item['expected_docs']}"
                    )

    n_facts = sum(
        len(json.loads(l).get("facts", []))
        for l in lines
        if l.strip() and not l.strip().startswith("#")
    )
    print(f"문항 {len(seen_ids)}개 / 팩트 {n_facts}개 검사")
    if errors:
        print(f"\n오류 {len(errors)}개:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("모두 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
