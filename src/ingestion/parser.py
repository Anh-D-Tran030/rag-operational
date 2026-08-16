"""Parse PDF and plain-text operational documents into page blocks."""
import re


async def parse_pdf(pdf_path: str) -> list[dict]:
    """Parse a PDF file into page-aware markdown blocks.

    Returns list of {"page_num": int, "content": str, "table_flag": bool}.
    Raises FileNotFoundError if pdf_path does not exist.
    """
    import os

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    import pymupdf4llm

    md_text = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    pages = []
    for i, page in enumerate(md_text):
        content = page["text"] if isinstance(page, dict) else str(page)
        table_flag = bool(re.search(r"\|[-:]+\|", content))
        pages.append({"page_num": i + 1, "content": content, "table_flag": table_flag})
    return pages


async def parse_text(text_path: str, doc_type: str = "shift_log") -> list[dict]:
    """Parse a plain-text file into logical blocks.

    Splits on double newlines. Returns list of {"page_num", "content", "table_flag", "doc_type"}.
    Raises FileNotFoundError if text_path does not exist.
    """
    import os

    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Text file not found: {text_path}")

    with open(text_path, encoding="utf-8") as f:
        raw = f.read()

    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
    if not blocks:
        blocks = [line.strip() for line in raw.splitlines() if line.strip()]

    return [
        {"page_num": i + 1, "content": block, "table_flag": False, "doc_type": doc_type}
        for i, block in enumerate(blocks)
    ]
