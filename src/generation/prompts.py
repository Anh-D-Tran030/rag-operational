"""Prompt templates for RAG generation and retrieval grading."""
from __future__ import annotations


def build_vector_rag_prompt(query: str, chunks: list[dict]) -> str:
    """Build a prompt grounded in retrieved document chunks.

    Instructs the LLM to answer only from provided context and cite
    doc_type, venue, date, and page_num for every fact used.
    """
    context_parts = []
    for i, c in enumerate(chunks, 1):
        meta = (
            f"[{i}] doc_type={c.get('doc_type','?')} venue={c.get('venue','?')} "
            f"date={c.get('date','?')} shift_id={c.get('shift_id','?')} "
            f"page={c.get('page_num','?')}"
        )
        context_parts.append(f"{meta}\n{c.get('content', '')}")

    context = "\n\n".join(context_parts)
    return (
        "Answer only from the provided context. "
        "Cite doc_type, venue, date, and page_num for each fact you use.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )


def build_hybrid_prompt(
    query: str,
    chunks: list[dict],
    sql_result: dict | None,
) -> str:
    """Build a prompt fusing vector chunks with SQL result rows.

    SQL result rows are formatted as a markdown table. Vector chunks are
    shown as cited context below.
    """
    parts: list[str] = [
        "Answer only from the provided context. Cite your sources.\n\n"
        f"Question: {query}\n"
    ]

    if sql_result and sql_result.get("rows"):
        rows: list[dict] = sql_result["rows"]
        headers = list(rows[0].keys()) if rows else []
        header_line = " | ".join(headers)
        sep_line = " | ".join(["---"] * len(headers))
        data_lines = [" | ".join(str(r.get(h, "")) for h in headers) for r in rows]
        table = "\n".join([header_line, sep_line] + data_lines)
        parts.append(f"## Database Result ({sql_result.get('view_name', 'SQL')})\n{table}\n")

    if chunks:
        context_parts = []
        for i, c in enumerate(chunks, 1):
            meta = (
                f"[{i}] doc_type={c.get('doc_type','?')} venue={c.get('venue','?')} "
                f"date={c.get('date','?')} page={c.get('page_num','?')}"
            )
            context_parts.append(f"{meta}\n{c.get('content', '')}")
        parts.append("## Document Context\n" + "\n\n".join(context_parts))

    parts.append("\nAnswer:")
    return "\n".join(parts)


def build_retrieval_grader_prompt(query: str, chunk: dict) -> str:
    """Build a yes/no grader prompt: is this chunk relevant to the query?"""
    return (
        f"Is the following document chunk relevant to answering the query?\n\n"
        f"Query: {query}\n\n"
        f"Chunk: {chunk.get('content', '')}\n\n"
        "Answer with only 'yes' or 'no'."
    )
