import hashlib
import json
from pathlib import Path

from regulations.chunker import chunk_markdown
from regulations.openai_client import embed_texts


def list_corpus_files(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.glob("*.md")
        if not p.name.startswith("_") and not p.name.startswith("~$")
    )


def compute_source_hash(md_files: list[Path]) -> str:
    sha = hashlib.sha256()
    for p in sorted(md_files, key=lambda x: x.name):
        sha.update(p.name.encode("utf-8"))
        sha.update(b"\n")
        sha.update(p.read_bytes())
    return sha.hexdigest()


def build_index(folder: Path, index_path: Path, force: bool = False) -> bool:
    files = list_corpus_files(folder)
    if not files:
        raise FileNotFoundError(f"No .md corpus files found in {folder}")
    current_hash = compute_source_hash(files)
    if not force and index_path.exists():
        try:
            old = json.loads(index_path.read_text(encoding="utf-8"))
            if old.get("source_hash") == current_hash:
                return False
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass  # corrupt index -> fall through to rebuild

    chunks = []
    for p in files:
        raw = p.read_text(encoding="utf-8")
        chunks.extend(chunk_markdown(raw, p.stem))

    embeddings = embed_texts([c["text"] for c in chunks])
    for c, emb in zip(chunks, embeddings):
        c["embedding"] = emb

    index_path.write_text(
        json.dumps({"source_hash": current_hash, "chunks": chunks}, ensure_ascii=False),
        encoding="utf-8",
    )
    return True
