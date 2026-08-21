"""검색 품질 점검용 스크립트 (봇과 무관, 필요할 때만 실행).

사용법:
    python _check_retrieval.py "What should I do if all courses are full?"

무엇을 보여주나:
  1) 영어 질문이 한국어 검색 키워드로 어떻게 번역됐는지
  2) 실제로 어떤 문서 청크가 상위로 잡혔는지 (유사도 점수 포함)

답이 코퍼스에 있는데도 엉뚱한 문서가 잡히면 검색 문제,
아예 관련 문서가 없으면 자료 문제다.
"""
import os
import sys

os.environ.setdefault("CORPUS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus"))

import numpy as np  # noqa: E402

import api  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    args = [a for a in sys.argv[1:] if a != "--dump"]
    question = " ".join(args)

    api._ensure_index(force_check=True)
    print(f"corpus chunks : {len(api.CHUNKS)}")

    print("\n[번역 호출 원본 확인]")
    try:
        raw = api.openai_chat(
            api.QUERY_EXPANSION_PROMPT, question, max_tokens=api.MAX_QUERY_TOKENS
        )
        print(f"  model={api.CHAT_MODEL} MAX_QUERY_TOKENS={api.MAX_QUERY_TOKENS}")
        print(f"  결과: {raw!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"  실패: {type(exc).__name__}: {exc}")

    queries = api.expand_queries(question)
    print(f"\n[질의 확장]")
    for i, q in enumerate(queries):
        print(f"  {i + 1}. {q}")

    matrix = np.array([c['embedding'] for c in api.CHUNKS], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    best = None
    for q in queries:
        v = np.array(api.embed_query(q), dtype=np.float32)
        sims = (matrix @ v) / (norms * np.linalg.norm(v) + 1e-9)
        best = sims if best is None else np.maximum(best, sims)

    print(f"\n[유사도 상위 10개]")
    for rank, i in enumerate(np.argsort(best)[::-1][:10], 1):
        c = api.CHUNKS[i]
        preview = c["text"].replace("\n", " ")[:70]
        print(f"  {rank:2d}. {best[i]:.3f}  {c['heading_path']}")
        print(f"      {preview}...")

    # 여기부터가 진짜 중요한 부분: 모델이 실제로 받는 컨텍스트.
    # 위 상위 10개와 다르다 (retrieve() 는 같은 문서의 나머지 조각을 채워 넣는다).
    contexts = api.retrieve(question)
    total = sum(len(c["text"]) for c in contexts)
    print(f"\n[모델에 실제로 들어가는 컨텍스트] {len(contexts)}개 청크 / {total:,}자 "
          f"(상한 {api.MAX_CONTEXT_CHARS:,})")
    by_source: dict[str, int] = {}
    for c in contexts:
        by_source[c["source"]] = by_source.get(c["source"], 0) + 1
    for src, n in by_source.items():
        print(f"  - {src}: {n}개 조각")

    joined = "\n".join(c["text"] for c in contexts)
    print("\n[핵심 사실이 컨텍스트에 들어갔는지]")
    for label, needle in [
        ("신청 시기(2월)", "2월"),
        ("신청 시기(8월)", "8월"),
        ("수령처(600주년기념관)", "600주년"),
        ("신청처(idcard)", "idcard"),
        ("증원 신청(책가방)", "책가방"),
        ("정원여석", "정원여석"),
    ]:
        print(f"  {'있음' if needle in joined else '없음'}  {label}")

    if "--dump" in sys.argv:
        print("\n[컨텍스트 전문]")
        for i, c in enumerate(contexts, 1):
            print(f"\n--- [{i}] {c['heading_path']} ---\n{c['text']}")


if __name__ == "__main__":
    main()
