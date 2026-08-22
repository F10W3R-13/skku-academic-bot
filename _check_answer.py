"""답변 생성까지 포함한 종단 점검 스크립트 (봇과 무관, 필요할 때만 실행).

사용법:
    python _check_answer.py "When and where can I get my student ID card?"

한 번의 실행에서 질의 분해 -> 확장 키워드 -> 컨텍스트 -> 핵심 사실 -> 답변까지
전부 보여준다. 분해/번역은 실행마다 달라지므로(모델 샘플링) 검색 점검과 답변
점검을 따로 돌리면 서로 다른 컨텍스트를 보고 있어 판단이 어긋난다 — 그래서
분해/번역/임베딩을 한 번만 하고 캐시로 재사용한다(화면 = 실제).

판독법:
    - 컨텍스트에 핵심 사실이 없다  -> 검색 문제 (키워드가 어디로 샜는지 본다)
    - 컨텍스트에 있는데 답변이 틀림 -> 생성 문제 (프롬프트/모델)
"""
import os
import sys

os.environ.setdefault("CORPUS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus"))

import api  # noqa: E402


# 벤치마크 질문(학생증) 기준 핵심 사실. 다른 질문을 볼 때는 출력을 참고용으로만.
NEEDLES = [
    ("신청 시기(2월)", "2월"),
    ("신청 시기(8월)", "8월"),
    ("수령처(600주년기념관)", "600주년"),
    ("신청처(idcard)", "idcard"),
    ("연락처(02-760-1077)", "02-760-1077"),
    ("즉시 발급(실전 메모)", "즉시 발급"),
]


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    args = [a for a in sys.argv[1:] if a != "--dump"]
    question = " ".join(args)

    api._ensure_index(force_check=True)

    # 분해와 번역은 여기서 딱 한 번 한다. 아래에서 retrieve() 가 같은 결과를
    # 쓰도록 함수를 캐시 버전으로 갈아끼운다 — 그래야 화면이 곧 실제다.
    # 주의: 원본을 먼저 포획한다. 안 그러면 패치 후 캐시 함수가 패치된 자신을
    # 다시 호출해 무한 재귀에 빠진다.
    parts = api.split_questions(question)
    original_expand = api.expand_queries
    original_embed = api.embed_query
    expand_cache: dict[str, list[str]] = {}
    embed_cache: dict[str, list[float]] = {}

    def cached_expand(q: str) -> list[str]:
        if q not in expand_cache:
            expand_cache[q] = original_expand(q)
        return expand_cache[q]

    def cached_embed(q: str) -> list[float]:
        if q not in embed_cache:
            embed_cache[q] = original_embed(q)
        return embed_cache[q]

    print(f"[질의 분해] model={api.CHAT_MODEL} MAX_QUERY_TOKENS={api.MAX_QUERY_TOKENS}")
    for i, p in enumerate(parts, 1):
        print(f"  파트 {i}: {p}")
        for kw in cached_expand(p)[1:]:
            print(f"      키워드: {kw}")

    api.split_questions = lambda q: parts   # 방금 본 파트를 그대로 쓰게 한다
    api.expand_queries = cached_expand      # 방금 본 번역을 그대로 쓰게 한다
    api.embed_query = cached_embed          # 방금 본 벡터를 그대로 쓰게 한다

    contexts = api.retrieve(question)
    total = sum(len(c["text"]) for c in contexts)
    print(f"\n[컨텍스트] {len(contexts)}개 청크 / {total:,}자 (상한 {api.MAX_CONTEXT_CHARS:,})")
    by_source: dict[str, int] = {}
    for c in contexts:
        by_source[c["source"]] = by_source.get(c["source"], 0) + 1
    for src, n in by_source.items():
        print(f"  - {src}: {n}개 조각")

    joined = "\n".join(c["text"] for c in contexts)
    print("\n[핵심 사실이 컨텍스트에 들어갔는지]")
    for label, needle in NEEDLES:
        print(f"  {'있음' if needle in joined else '없음'}  {label}")

    if "--dump" in sys.argv:
        print("\n[컨텍스트 전문]")
        for i, c in enumerate(contexts, 1):
            print(f"\n--- [{i}] {c['heading_path']} ---\n{c['text']}")

    answer, sources = api.generate_answer(question, contexts)
    print("\n[답변]")
    print(answer)
    print(f"\n[sources] {', '.join(sources)}")


if __name__ == "__main__":
    main()
