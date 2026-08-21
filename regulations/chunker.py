import re

MAX_CHARS = 1500


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            return text[end + 3:].lstrip("\n")
    return text


def _split_long(text: str, limit: int = MAX_CHARS) -> list[str]:
    parts: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n", text):
        while len(para) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(para[:limit])
            para = para[limit:]
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) > limit and current:
            parts.append(current)
            current = para
        else:
            current = candidate
    if current.strip():
        parts.append(current)
    return parts


def chunk_markdown(raw_text: str, source_title: str) -> list[dict]:
    body = _strip_frontmatter(raw_text)
    if not body.strip():
        return []
    lines = body.splitlines()
    chunks: list[dict] = []
    h2, h3 = "", ""
    buf: list[str] = []

    def flush():
        text = "\n".join(buf).strip()
        buf.clear()
        if not text:
            return
        path = " > ".join(p for p in (source_title, h2, h3) if p)
        for piece in _split_long(text):
            chunks.append({"source": source_title, "heading_path": path, "text": piece})

    for line in lines:
        m2 = re.match(r"^##\s+(?!#)(.+)", line)
        m3 = re.match(r"^###\s+(.+)", line)
        if m2:
            flush()
            h2, h3 = m2.group(1).strip(), ""
        elif m3:
            flush()
            h3 = m3.group(1).strip()
        else:
            buf.append(line)
    flush()
    return chunks
