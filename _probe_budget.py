"""답변 생성 호출의 토큰 사용량을 실측하는 스크립트 (예산 튜닝용).

사용법:
    python _probe_budget.py "질문..." [예산]

추론 모델은 max_completion_tokens 예산에서 추론 토큰을 먼저 쓴다. 추론에 몇 토큰을
쓰는지 재야 답변 예산(MAX_ANSWER_TOKENS)을 근거 있게 정할 수 있다.
"""
import os
import sys

os.environ.setdefault("CORPUS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus"))

import api  # noqa: E402
from regulations.openai_client import get_client  # noqa: E402


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.isdigit()]
    budget = next((int(a) for a in sys.argv[1:] if a.isdigit()), 2000)
    if not args:
        print(__doc__)
        raise SystemExit(1)
    question = " ".join(args)

    api._ensure_index(force_check=True)
    contexts = api.retrieve(question)
    blocks = "\n\n".join(
        f"[{i + 1}] Document: {c['source']} | Section: {c['heading_path']}\n{c['text']}"
        for i, c in enumerate(contexts)
    )
    messages = [
        {"role": "system", "content": api.SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{blocks}"},
    ]
    resp = get_client().chat.completions.create(
        model=api.CHAT_MODEL, messages=messages, max_completion_tokens=budget
    )
    ch = resp.choices[0]
    u = resp.usage
    content = (ch.message.content or "").strip()
    print(f"budget            : {budget}")
    print(f"finish_reason     : {ch.finish_reason}")
    print(f"prompt_tokens     : {u.prompt_tokens}")
    print(f"completion_tokens : {u.completion_tokens}")
    det = getattr(u, "completion_tokens_details", None)
    print(f"reasoning_tokens  : {getattr(det, 'reasoning_tokens', '?')}")
    print(f"visible_chars     : {len(content)}")
    print("---")
    print(content[:400])


if __name__ == "__main__":
    main()
