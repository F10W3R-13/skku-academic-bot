import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 모델은 .env 로 바꿀 수 있다 (코드 수정 없이 교체/롤백 가능).
#   EMBED_MODEL 을 바꾸면 인덱스가 자동으로 전체 재빌드된다(차원이 달라지므로).
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.6-luna")


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in bot/.env")
    return OpenAI()


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = get_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend(d.embedding for d in resp.data)
    return out
