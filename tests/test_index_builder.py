import hashlib
from pathlib import Path

from regulations.index_builder import compute_source_hash, list_corpus_files


def test_list_corpus_files_skips_underscore(tmp_path: Path):
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "_skip.md").write_text("s", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    names = [p.name for p in list_corpus_files(tmp_path)]
    assert names == ["a.md"]


def test_source_hash_changes_when_content_changes(tmp_path: Path):
    f = tmp_path / "a.md"
    f.write_text("v1", encoding="utf-8")
    h1 = compute_source_hash([f])
    f.write_text("v2", encoding="utf-8")
    h2 = compute_source_hash([f])
    assert h1 != h2
    assert h1 == hashlib.sha256(b"a.md\nv1").hexdigest()


def test_source_hash_order_independent(tmp_path: Path):
    a = tmp_path / "a.md"; b = tmp_path / "b.md"
    a.write_text("A", encoding="utf-8"); b.write_text("B", encoding="utf-8")
    assert compute_source_hash([a, b]) == compute_source_hash([b, a])


def test_corrupt_index_file_triggers_rebuild(tmp_path: Path, monkeypatch):
    import regulations.index_builder as ib
    monkeypatch.setattr(ib, "embed_texts", lambda texts: [[0.0] * 4 for _ in texts])
    folder = tmp_path / "corpus"
    folder.mkdir()
    (folder / "a.md").write_text("# A\n\ntext", encoding="utf-8")
    index_path = tmp_path / "index.json"
    index_path.write_text("{not valid json", encoding="utf-8")
    assert ib.build_index(folder, index_path) is True
