"""Split page blocks into token-aware chunks respecting page boundaries."""


def chunk_pages(pages: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """Split each page's content into overlapping chunks.

    Table pages (table_flag=True) are kept as a single chunk.
    Chunks never merge content across page boundaries.
    Each output chunk: {"content", "page_num", "table_flag", "chunk_index", "doc_type"}.
    """
    chunks: list[dict] = []
    chunk_index = 0

    for page in pages:
        content: str = page["content"]
        page_num: int = page["page_num"]
        table_flag: bool = page.get("table_flag", False)
        doc_type: str = page.get("doc_type", "unknown")

        if table_flag:
            chunks.append(
                {
                    "content": content,
                    "page_num": page_num,
                    "table_flag": table_flag,
                    "chunk_index": chunk_index,
                    "doc_type": doc_type,
                }
            )
            chunk_index += 1
            continue

        words = content.split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunks.append(
                {
                    "content": " ".join(chunk_words),
                    "page_num": page_num,
                    "table_flag": False,
                    "chunk_index": chunk_index,
                    "doc_type": doc_type,
                }
            )
            chunk_index += 1
            if end >= len(words):
                break
            start = end - overlap

    return chunks
