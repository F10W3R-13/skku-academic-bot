"""모델이 어떤 호출 파라미터를 실제로 받아주는지 확인하는 스크립트.

openai_chat() 은 temperature 거부를 가정하고 파라미터를 하나씩 빼며 재시도한다.
어떤 시도가 성공하는지 모르면(=온도가 실제로 적용되는지 모르면) 번역 분산 문제를
잘못 고칠 수 있다. 이 스크립트는 그 가정을 직접 검증한다.
"""
import os
import sys

os.environ.setdefault("CORPUS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus"))

from regulations.openai_client import CHAT_MODEL, get_client  # noqa: E402


def try_call(label: str, **kwargs) -> bool:
    try:
        resp = get_client().chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": "Say OK."}],
            **kwargs,
        )
        content = (resp.choices[0].message.content or "").strip()
        print(f"[OK]   {label} -> {content!r}")
        return True
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).replace("\n", " ")[:180]
        print(f"[FAIL] {label} -> {type(exc).__name__}: {msg}")
        return False


def main() -> None:
    print(f"model = {CHAT_MODEL}\n")
    try_call("max_tokens=50", max_tokens=50)
    try_call("max_completion_tokens=50", max_completion_tokens=50)
    try_call("temperature=0.2 + max_completion_tokens=50",
             temperature=0.2, max_completion_tokens=50)
    try_call("seed=42 + temperature=0.2 + max_completion_tokens=50",
             seed=42, temperature=0.2, max_completion_tokens=50)
    try_call("response_format=json_object", response_format={"type": "json_object"},
             max_completion_tokens=200)


if __name__ == "__main__":
    main()
