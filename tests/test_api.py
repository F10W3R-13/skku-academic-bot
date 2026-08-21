import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture()
def client(monkeypatch):
    api.CHUNKS = [
        {"source": "졸업요건", "heading_path": "졸업요건 > 학점", "text": "총 140학점",
         "embedding": [1.0, 0.0]},
        {"source": "등록절차", "heading_path": "등록절차 > 일정", "text": "2월 납부",
         "embedding": [0.0, 1.0]},
    ]
    monkeypatch.setattr(api, "embed_query", lambda q: [1.0, 0.0])
    monkeypatch.setattr(api, "expand_queries", lambda q: [q])
    monkeypatch.setattr(api, "generate_answer", lambda q, ctx: ("You need 140 credits.", ["졸업요건"]))
    return TestClient(api.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_and_sources(client):
    r = client.post("/ask", json={"question": "How many credits to graduate?"})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "You need 140 credits."
    assert data["sources"] == ["졸업요건"]


def test_retrieve_orders_by_similarity(client):
    ctx = api.retrieve("credits", k=2)
    assert ctx[0]["source"] == "졸업요건"


def test_ask_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_corpus_dir_defaults_to_bot_corpus_not_parent(monkeypatch):
    """개인 성적/학적 파일이 있는 상위 폴더를 기본 코퍼스로 잡으면 안 된다."""
    import importlib

    monkeypatch.delenv("CORPUS_DIR", raising=False)
    reloaded = importlib.reload(api)
    try:
        assert reloaded.CORPUS_DIR == reloaded.BOT_DIR / "corpus"
        assert reloaded.CORPUS_DIR != reloaded.BOT_DIR.parent
    finally:
        importlib.reload(api)


def test_corpus_dir_env_override_still_works(monkeypatch, tmp_path):
    import importlib

    monkeypatch.setenv("CORPUS_DIR", str(tmp_path))
    reloaded = importlib.reload(api)
    try:
        assert reloaded.CORPUS_DIR == tmp_path
    finally:
        monkeypatch.delenv("CORPUS_DIR", raising=False)
        importlib.reload(api)


def test_expand_queries_adds_korean_query(monkeypatch):
    monkeypatch.setattr(
        api, "openai_chat", lambda system, user, max_tokens=None: "수강신청 정정 증원 여석"
    )
    qs = api.expand_queries("What if all courses are full?")
    assert qs[0] == "What if all courses are full?"
    assert "수강신청" in qs[1]


def test_expand_queries_falls_back_when_translation_fails(monkeypatch):
    def boom(system, user, max_tokens=None):
        raise RuntimeError("no api key")

    monkeypatch.setattr(api, "openai_chat", boom)
    assert api.expand_queries("hello") == ["hello"]


def test_retrieve_uses_best_score_across_queries(monkeypatch):
    api.CHUNKS = [
        {"source": "english_doc", "heading_path": "a", "text": "dorm", "embedding": [1.0, 0.0]},
        {"source": "korean_doc", "heading_path": "b", "text": "수강신청", "embedding": [0.0, 1.0]},
    ]
    api._MATRIX = None
    # 영어 질의는 english_doc 쪽, 한국어 확장 질의는 korean_doc 쪽을 가리킨다.
    monkeypatch.setattr(api, "expand_queries", lambda q: ["english", "한국어"])
    monkeypatch.setattr(
        api, "embed_query", lambda q: [1.0, 0.0] if q == "english" else [0.0, 1.0]
    )
    sources = [c["source"] for c in api.retrieve("q", k=2)]
    assert set(sources) == {"english_doc", "korean_doc"}


def test_retrieve_returns_empty_when_index_empty(monkeypatch):
    api.CHUNKS = []
    api._MATRIX = None
    monkeypatch.setattr(api, "expand_queries", lambda q: [q])
    assert api.retrieve("anything") == []


def test_openai_chat_retries_when_max_tokens_unsupported(monkeypatch):
    calls = []

    class FakeResp:
        class _C:
            class _M:
                content = " ok "
            message = _M()
        choices = [_C()]

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    calls.append(kwargs)
                    if "max_tokens" in kwargs:
                        raise RuntimeError("Unsupported parameter: 'max_tokens'")
                    return FakeResp()

    monkeypatch.setattr("regulations.openai_client.get_client", lambda: FakeClient)
    assert api.openai_chat("sys", "user") == "ok"
    assert "max_tokens" in calls[0]
    assert "max_completion_tokens" in calls[1]


def test_openai_chat_does_not_swallow_real_errors(monkeypatch):
    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("insufficient_quota")

    monkeypatch.setattr("regulations.openai_client.get_client", lambda: FakeClient)
    with pytest.raises(RuntimeError, match="insufficient_quota"):
        api.openai_chat("sys", "user")


def _chunk(source, idx, text, emb):
    return {"source": source, "heading_path": f"{source} > {idx}", "text": text, "embedding": emb}


def test_retrieve_fills_in_sibling_chunks_of_the_same_document(monkeypatch):
    """한 문서가 여러 조각으로 갈려도 같은 문서의 나머지 조각이 함께 들어와야 한다."""
    api.CHUNKS = [
        _chunk("학생증", 1, "신청 시기는 2월 말", [1.0, 0.0]),
        _chunk("학생증", 2, "수령처는 600주년기념관 1층", [0.6, 0.8]),
        _chunk("학생증", 3, "재발급 7,000원", [0.5, 0.86]),
        _chunk("무관문서", 1, "관계없는 내용", [0.0, 1.0]),
    ]
    api._MATRIX = None
    monkeypatch.setattr(api, "expand_queries", lambda q: [q])
    monkeypatch.setattr(api, "embed_query", lambda q: [1.0, 0.0])

    got = api.retrieve("어디서 받나요?", k=1)
    sources = [c["source"] for c in got]
    assert sources[:3] == ["학생증"] * 3, "같은 문서의 나머지 조각이 따라 들어와야 함"
    assert "600주년기념관" in " ".join(c["text"] for c in got)


def test_retrieve_respects_context_budget(monkeypatch):
    """큰 문서가 컨텍스트를 통째로 먹지 않아야 한다."""
    api.CHUNKS = [_chunk("큰문서", i, "가" * 400, [1.0, 0.0]) for i in range(50)]
    api._MATRIX = None
    monkeypatch.setattr(api, "expand_queries", lambda q: [q])
    monkeypatch.setattr(api, "embed_query", lambda q: [1.0, 0.0])
    monkeypatch.setattr(api, "MAX_CONTEXT_CHARS", 2000)

    got = api.retrieve("q", k=1)
    total = sum(len(c["text"]) for c in got)
    assert total <= 2000
    assert len(got) == 5


def test_retrieve_keeps_document_order_within_a_source(monkeypatch):
    """조각들이 원문 순서대로 읽혀야 모델이 절차를 제대로 이해한다."""
    api.CHUNKS = [
        _chunk("문서", 0, "1단계", [1.0, 0.0]),
        _chunk("문서", 1, "2단계", [0.9, 0.1]),
        _chunk("문서", 2, "3단계", [0.95, 0.05]),
    ]
    api._MATRIX = None
    monkeypatch.setattr(api, "expand_queries", lambda q: [q])
    monkeypatch.setattr(api, "embed_query", lambda q: [1.0, 0.0])

    got = api.retrieve("q", k=1)
    assert [c["text"] for c in got] == ["1단계", "2단계", "3단계"]


def test_expand_queries_reports_empty_translation(monkeypatch, capsys):
    """추론 모델이 예산 부족으로 빈 응답을 줄 때, 조용히 넘어가지 말고 이유를 남겨야 한다."""
    monkeypatch.setattr(api, "openai_chat", lambda system, user, max_tokens=None: "")
    assert api.expand_queries("hello") == ["hello"]
    assert "비어 있음" in capsys.readouterr().err


def test_expand_queries_reports_failure_reason(monkeypatch, capsys):
    def boom(system, user, max_tokens=None):
        raise RuntimeError("model_not_found")

    monkeypatch.setattr(api, "openai_chat", boom)
    assert api.expand_queries("hello") == ["hello"]
    assert "model_not_found" in capsys.readouterr().err


def test_openai_chat_handles_none_content(monkeypatch):
    class FakeResp:
        class _C:
            class _M:
                content = None
            message = _M()
        choices = [_C()]

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return FakeResp()

    monkeypatch.setattr("regulations.openai_client.get_client", lambda: FakeClient)
    assert api.openai_chat("sys", "user") == ""


def test_retrieve_does_not_fill_loosely_related_documents(monkeypatch):
    """애매하게 걸린 문서까지 통째로 넣으면 정작 중요한 문서의 신호가 희석된다."""
    api.CHUNKS = (
        [_chunk("정답문서", i, f"핵심 사실 {i}", [1.0, 0.0]) for i in range(5)]
        + [_chunk("애매문서", i, f"곁다리 {i}", [0.55, 0.84]) for i in range(20)]
    )
    api._MATRIX = None
    monkeypatch.setattr(api, "expand_queries", lambda q: [q])
    monkeypatch.setattr(api, "embed_query", lambda q: [1.0, 0.0])

    got = api.retrieve("q", k=8)
    counts: dict[str, int] = {}
    for c in got:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    assert counts["정답문서"] == 5, "1위 문서는 조각이 모두 채워져야 함"
    assert counts.get("애매문서", 0) <= 8, "점수 낮은 문서까지 통째로 들어오면 안 됨"


def test_retrieve_fills_second_document_when_scores_are_close(monkeypatch):
    """점수가 비등한 문서는 함께 채운다."""
    api.CHUNKS = (
        [_chunk("A문서", i, f"A{i}", [1.0, 0.0]) for i in range(3)]
        + [_chunk("B문서", i, f"B{i}", [0.999, 0.045]) for i in range(3)]
    )
    api._MATRIX = None
    monkeypatch.setattr(api, "expand_queries", lambda q: [q])
    monkeypatch.setattr(api, "embed_query", lambda q: [1.0, 0.0])

    # k=4 여야 두 문서가 모두 씨앗에 들어간다 (씨앗에 없는 문서는 채우지 않는다)
    got = api.retrieve("q", k=4)
    counts: dict[str, int] = {}
    for c in got:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    assert counts == {"A문서": 3, "B문서": 3}
