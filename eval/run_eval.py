"""골드셋 배치 평가: 검색 → 생성 → 1층 자동 채점 → 결과 JSON.

사용법 (bot/ 디렉터리에서 실행):
    python eval/run_eval.py                       # 전체 문항
    python eval/run_eval.py --limit 3             # 앞 3문항 스모크 테스트
    python eval/run_eval.py --only single,multi   # 카테고리 필터
    python eval/run_eval.py --ids g1-001,robust-002
    python eval/run_eval.py --skip-absent         # absent/adv 제외(비용 절약)

api.retrieve / generate_answer 을 직접 호출한다(/ask HTTP 경유가 아니라).
컨텍스트와 문서별 검색 유무까지 기록할 수 있어 실패 원인 분류가 가능하기 때문.

1층 자동 채점 (결정론적):
    docs_retrieved - expected_docs 각 문서가 컨텍스트에 있는가
    fact ctx_hit   - 각 fact의 context_any가 컨텍스트 원문에 있는가
    fact ans_hit   - 각 fact의 answer_any가 답변에 있는가
    ASCII needle은 단어 경계 매칭 — '2'가 '24'에, '130'이 '1300'에 맞는 것 방지.

triage (문항별 실패 원인):
    PASS            전부 통과
    RETRIEVAL_MISS  정답 문서(일부)가 컨텍스트에 없음      → 검색 문제
    CONTEXT_MISS    문서는 왔으나 핵심 사실 절이 잘림      → 컨텍스트 예산/fill 문제
    GENERATION_MISS 사실이 컨텍스트에 있는데 답변이 놓침    → 프롬프트/모델 문제
    JUDGE           absent/adversarial — 자동 채점 불가, 2층 심판 판정
    ERROR           예외 발생
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BOT_DIR = EVAL_DIR.parent
sys.path.insert(0, str(BOT_DIR))

import os  # noqa: E402

os.environ.setdefault("CORPUS_DIR", str(BOT_DIR / "corpus"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BOT_DIR / ".env")  # api 임포트 전에 명시적으로 로드

import api  # noqa: E402

QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
AUTO_CATEGORIES = {"single", "multi", "robust"}  # 자동 채점 대상

_ASCII_NEEDLE = re.compile(r"^[\x21-\x7E]+$")


def _match(text: str, needle: str) -> bool:
    """needle이 text에 있는가. ASCII needle은 단어 경계로.

    양쪽 다 casefold해서 비교한다. 여러 단어로 된 needle은 개행·불릿 같은
    서식 때문에 어긋나는 경우가 있어(실측: 'Flights, transportation'이
    불릿 목록 'Flights\\nTransportation'에 안 걸림) 공백을 통합한
    텍스트에서도 한 번 더 찾는다.
    """
    hay = text.casefold()
    n = needle.casefold()
    if _ASCII_NEEDLE.match(needle):
        pattern = rf"(?<![0-9A-Za-z]){re.escape(n)}(?![0-9A-Za-z])"
        if re.search(pattern, hay):
            return True
    if n in hay:
        return True
    if " " in n.strip():
        # 콜론/대시/불릿 같은 서식을 양쪽에서 벗겨 단어열로 비교한다
        # (실측: 'Monday: 13:00' 답변에 needle 'Monday 13:00'이 안 걸림).
        strip = lambda s: re.sub(r"\s+", " ", re.sub(r"[\*\-•·:]", " ", s))
        return strip(n) in strip(hay)
    return False


def _any_match(text: str, needles: list[str]) -> bool:
    return any(_match(text, x) for x in needles)


def load_questions() -> list[dict]:
    items = []
    for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def eval_one(item: dict) -> dict:
    rec: dict = {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "expected_docs": item.get("expected_docs", []),
    }
    t0 = time.time()
    try:
        contexts = api.retrieve(item["question"])
        answer, sources = api.generate_answer(item["question"], contexts)
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "ERROR"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec
    rec["latency_s"] = round(time.time() - t0, 1)

    if item["category"] not in AUTO_CATEGORIES:
        # absent / adversarial: 답변만 수집해 2층 심판(judge)이 판정한다.
        rec["status"] = "JUDGE"
        rec["answer"] = answer
        rec["retrieved_sources"] = list(dict.fromkeys(c["source"] for c in contexts))
        return rec

    rec["answer"] = answer
    rec["retrieved_sources"] = list(dict.fromkeys(c["source"] for c in contexts))
    rec["context_chunks"] = [
        {"source": c["source"], "heading": c["heading_path"]} for c in contexts
    ]
    rec["context_chars"] = sum(len(c["text"]) for c in contexts)
    # 오프라인 재채점(rescore.py)이 답변뿐 아니라 컨텍스트 검증도 다시 할 수
    # 있게 원문을 결과에 남긴다.
    rec["context_text"] = "\n".join(c["text"] for c in contexts)

    expected = [d.removesuffix(".md").casefold() for d in item.get("expected_docs", [])]
    retrieved = [s.casefold() for s in rec["retrieved_sources"]]
    rec["docs_retrieved"] = {
        doc: doc in retrieved for doc in expected
    }
    retrieval_ok = all(rec["docs_retrieved"].values()) if expected else True

    fact_results = []
    for f in item.get("facts", []):
        ctx_hit = _any_match(rec["context_text"], f["context_any"])
        ans_hit = _any_match(answer, f["answer_any"])
        fact_results.append(
            {"label": f["label"], "ctx_hit": ctx_hit, "ans_hit": ans_hit}
        )
    rec["facts"] = fact_results

    if not retrieval_ok:
        rec["status"] = "RETRIEVAL_MISS"
    elif fact_results and not all(fr["ctx_hit"] for fr in fact_results):
        rec["status"] = "CONTEXT_MISS"
    elif fact_results and not all(fr["ans_hit"] for fr in fact_results):
        rec["status"] = "GENERATION_MISS"
    else:
        rec["status"] = "PASS"
    return rec


def summarize(results: list[dict]) -> None:
    by_status: dict[str, list[str]] = {}
    for r in results:
        by_status.setdefault(r["status"], []).append(r["id"])

    print("\n" + "=" * 62)
    print(f"[요약] 문항 {len(results)}개")
    for status in ["PASS", "RETRIEVAL_MISS", "CONTEXT_MISS", "GENERATION_MISS", "ERROR", "JUDGE"]:
        ids = by_status.get(status, [])
        if ids:
            print(f"  {status:<15} {len(ids):3d}  ({', '.join(ids[:8])}{'…' if len(ids) > 8 else ''})")

    # 카테고리별 PASS율 (자동 채점 대상만)
    print("\n[카테고리별 PASS]")
    cats: dict[str, list[dict]] = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)
    for cat, rs in cats.items():
        auto = [r for r in rs if r["status"] != "JUDGE"]
        if auto:
            passed = sum(1 for r in auto if r["status"] == "PASS")
            print(f"  {cat:<12} {passed}/{len(auto)}")
        else:
            print(f"  {cat:<12} (심판 채점 대상 {len(rs)}개)")

    # 팩트 단위 지표
    facts = [f for r in results for f in r.get("facts", [])]
    if facts:
        ctx_hits = sum(1 for f in facts if f["ctx_hit"])
        ans_hits = sum(1 for f in facts if f["ans_hit"])
        n = len(facts)
        print(f"\n[팩트 단위] 총 {n}개")
        print(f"  컨텍스트 포함 {ctx_hits}/{n} ({ctx_hits / n:.0%})")
        print(f"  답변 포함     {ans_hits}/{n} ({ans_hits / n:.0%})")

    lat = [r["latency_s"] for r in results if "latency_s" in r]
    if lat:
        lat.sort()
        print(f"\n[지연] 평균 {sum(lat) / len(lat):.1f}s / 중앙 {lat[len(lat) // 2]:.1f}s / 최대 {lat[-1]:.1f}s")
    lens = [len(r["answer"]) for r in results if "answer" in r]
    if lens:
        lens.sort()
        print(f"[답변 길이] 평균 {sum(lens) / len(lens):,.0f}자 / 중앙 {lens[len(lens) // 2]:,}자 / 최대 {lens[-1]:,}자")
    csize = [r["context_chars"] for r in results if "context_chars" in r]
    if csize:
        csize.sort()
        print(f"[컨텍스트] 평균 {sum(csize) / len(csize):,.0f}자 / 중앙 {csize[len(csize) // 2]:,}자 / 최대 {csize[-1]:,}자")
    print("=" * 62)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="앞 N문항만 실행")
    parser.add_argument("--only", help="카테고리 콤마 목록 (예: single,multi)")
    parser.add_argument("--ids", help="특정 id 콤마 목록")
    parser.add_argument("--skip-absent", action="store_true", help="absent/adversarial 제외")
    parser.add_argument("--out", help="결과 파일 경로(기본: results/자동 생성)")
    args = parser.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    items = load_questions()
    if args.only:
        keep = {c.strip() for c in args.only.split(",")}
        items = [i for i in items if i["category"] in keep]
    if args.skip_absent:
        items = [i for i in items if i["category"] in AUTO_CATEGORIES]
    if args.ids:
        keep = {i.strip() for i in args.ids.split(",")}
        items = [i for i in items if i["id"] in keep]
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("실행할 문항이 없습니다 — 필터를 확인하세요.")

    api._ensure_index(force_check=True)
    print(f"[준비] 문항 {len(items)}개 / 청크 {len(api.CHUNKS)}개 / 모델 {api.CHAT_MODEL}")
    print(f"{'컨텍스트 상한':<20} {api.MAX_CONTEXT_CHARS:,}자")
    print(f"{'FILL_MIN_SCORE':<20} {api.FILL_MIN_SCORE}")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"eval_{stamp}.json"
    partial_path = out_path.with_suffix(".jsonl")  # 진행 상황(중단 대비)

    results = []
    t_start = time.time()
    for i, item in enumerate(items, 1):
        rec = eval_one(item)
        results.append(rec)
        partial_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n",
            encoding="utf-8",
        )
        mark = {
            "PASS": "✓", "RETRIEVAL_MISS": "✗검색", "CONTEXT_MISS": "✗절누락",
            "GENERATION_MISS": "✗생성", "ERROR": "!!", "JUDGE": "?심판",
        }[rec["status"]]
        print(f"[{i:3d}/{len(items)}] {item['id']:<14} {mark}", flush=True)

    meta = {
        "ran_at": stamp,
        "chat_model": api.CHAT_MODEL,
        "embed_model": api.__dict__.get("EMBED_MODEL", ""),
        "chunks": len(api.CHUNKS),
        "params": {
            k: getattr(api, k)
            for k in [
                "MAX_CONTEXT_CHARS", "MAX_FILL_SOURCES", "FILL_MIN_SCORE",
                "FILL_SCORE_MARGIN", "MIN_SEED_SCORE", "MAX_SEEDS_PER_SOURCE",
                "MAX_FILL_CHUNKS_PER_SOURCE", "TITLE_BONUS", "TITLE_CONCENTRATION",
                "MAX_ANSWER_TOKENS", "MAX_QUERY_TOKENS",
            ]
            if hasattr(api, k)
        },
        "elapsed_s": round(time.time() - t_start, 1),
        "counts": {},
    }
    for r in results:
        meta["counts"][r["status"]] = meta["counts"].get(r["status"], 0) + 1

    out_path.write_text(
        json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    partial_path.unlink(missing_ok=True)
    print(f"\n[저장] {out_path}")
    summarize(results)


if __name__ == "__main__":
    main()
