"""Unit tests for src/generation/prompts.py."""
from src.generation.prompts import (
    build_hybrid_prompt,
    build_retrieval_grader_prompt,
    build_vector_rag_prompt,
)


def test_vector_prompt_has_doc_type():
    chunks = [{
        "doc_type": "compliance", "content": "Staff must be 5.",
        "page_num": 3, "venue": "RSL", "date": "2026-01-01", "shift_id": None,
    }]
    prompt = build_vector_rag_prompt("What is the requirement?", chunks)
    assert "compliance" in prompt
    assert "Staff must be 5." in prompt


def test_vector_prompt_instructs_citation():
    chunks = [{
        "doc_type": "sop", "content": "Evacuate immediately.",
        "page_num": 1, "venue": "RSL", "date": "2026-01-01", "shift_id": None,
    }]
    prompt = build_vector_rag_prompt("What to do in emergency?", chunks)
    assert "doc_type" in prompt
    assert "page_num" in prompt


def test_hybrid_prompt_has_sql_table():
    sql_result = {
        "rows": [{"venue": "RSL", "staff_count": 12}, {"venue": "RSL", "staff_count": 8}],
        "view_name": "v_shift_summary",
        "sql_used": "SELECT venue, staff_count FROM v_shift_summary",
    }
    chunks = []
    prompt = build_hybrid_prompt("How many staff?", chunks, sql_result)
    assert "venue" in prompt
    assert "staff_count" in prompt
    assert "---" in prompt  # markdown table separator


def test_hybrid_prompt_no_sql():
    chunks = [{
        "doc_type": "shift_log", "content": "Busy night.",
        "page_num": 1, "venue": "RSL", "date": "2026-05-01", "shift_id": "SH-001",
    }]
    prompt = build_hybrid_prompt("What happened?", chunks, None)
    assert "Busy night." in prompt


def test_grader_prompt_is_yes_no():
    chunk = {"content": "The minimum staff requirement is 5."}
    prompt = build_retrieval_grader_prompt("What is the minimum staffing?", chunk)
    assert "yes" in prompt.lower() or "no" in prompt.lower()
    assert "minimum" in prompt or "staff" in prompt
