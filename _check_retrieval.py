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
    question = " ".join(sys.argv[1:])

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

    print(f"\n[상위 10개 청크]")
    for rank, i in enumerate(np.argsort(best)[::-1][:10], 1):
        c = api.CHUNKS[i]
        preview = c["text"].replace("\n", " ")[:70]
        print(f"  {rank:2d}. {best[i]:.3f}  {c['heading_path']}")
        print(f"      {preview}...")


if __name__ == "__main__":
    main()
