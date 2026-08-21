from regulations.chunker import chunk_markdown

SAMPLE = """---
title: 학사일정표
source_pdf: x.pdf
---

# 학사일정표

## 1학기 일정

### 수강신청

- 1월 말 수강신청
- 정정은 2월 초

### 개강

- 3월 2일 개강한다. 아주 긴 문단이 이어진다. """ + ("가" * 1600) + """

## 등록

등록금은 2월에 납부한다.
"""


def test_frontmatter_is_skipped():
    chunks = chunk_markdown(SAMPLE, "학사일정표")
    assert all("source_pdf" not in c["text"] for c in chunks)


def test_heading_paths_are_joined():
    chunks = chunk_markdown(SAMPLE, "학사일정표")
    paths = [c["heading_path"] for c in chunks]
    assert "학사일정표 > 1학기 일정 > 수강신청" in paths
    assert "학사일정표 > 등록" in paths


def test_every_chunk_has_source_and_text():
    chunks = chunk_markdown(SAMPLE, "학사일정표")
    assert len(chunks) >= 3
    for c in chunks:
        assert c["source"] == "학사일정표"
        assert c["text"].strip()


def test_long_sections_are_split():
    chunks = chunk_markdown(SAMPLE, "학사일정표")
    assert all(len(c["text"]) <= 1500 for c in chunks)


def test_empty_input():
    assert chunk_markdown("", "x") == []
