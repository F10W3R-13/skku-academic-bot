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
