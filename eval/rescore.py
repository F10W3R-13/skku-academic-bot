"""저장된 평가 결과를 API 호출 없이 재채점 (needle/매처 수정 후용).

run_eval.py 실행 당시의 답변·컨텍스트 원문을 결과 JSON에서 다시 읽어
facts 채점과 triage만 갱신한다. questions.jsonl 의 needle을 고쳤거나
_match() 매처를 고쳤을 때 유용 — 전체 재실행(비용/수십 분)이 필요 없다.

제약: 첫 실행 결과에 context_text 가 없으면(구버전 run_eval) 컨텍스트
채점(ctx_hit)은 옛 값 그대로 둔다.

사용법:
    python eval/rescore.py results/eval_20260822_115459.json
"""
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from run_eval import AUTO_CATEGORIES, _any_match, load_questions, summarize  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    result_path = Path(sys.argv[1])
    data = json.loads(result_path.read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in load_questions()}

    for r in data["results"]:
        item = questions.get(r["id"])
        if not item or r["status"] in ("JUDGE", "ERROR"):
            continue
        if item["category"] not in AUTO_CATEGORIES:
            continue

        answer = r.get("answer", "")
        context = r.get("context_text", "")

        expected = [d.removesuffix(".md").casefold() for d in item.get("expected_docs", [])]
        retrieved = [s.casefold() for s in r.get("retrieved_sources", [])]
        r["docs_retrieved"] = {doc: doc in retrieved for doc in expected}
        retrieval_ok = all(r["docs_retrieved"].values()) if expected else True

        new_facts = []
        for f in item.get("facts", []):
            ctx_hit = _any_match(context, f["context_any"]) if context else None
            if ctx_hit is None:  # 컨텍스트 원문이 없으면 옛 판정 유지
                old = next((x for x in r.get("facts", []) if x["label"] == f["label"]), {})
                ctx_hit = old.get("ctx_hit", False)
            new_facts.append(
                {"label": f["label"], "ctx_hit": ctx_hit, "ans_hit": _any_match(answer, f["answer_any"])}
            )
        r["facts"] = new_facts

        if not retrieval_ok:
            r["status"] = "RETRIEVAL_MISS"
        elif new_facts and not all(f["ctx_hit"] for f in new_facts):
            r["status"] = "CONTEXT_MISS"
        elif new_facts and not all(f["ans_hit"] for f in new_facts):
            r["status"] = "GENERATION_MISS"
        else:
            r["status"] = "PASS"

    counts = {}
    for r in data["results"]:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    data["meta"]["counts"] = counts
    data["meta"]["rescored"] = True

    out = result_path.with_name(result_path.stem + "_rescored.json")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[저장] {out}")
    summarize(data["results"])


if __name__ == "__main__":
    main()
