"""Streamlit UI for the Operational RAG API."""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Operational RAG", layout="wide")
st.title("Operational RAG Assistant")

# ── Session state ──────────────────────────────────────────────────────────────
if "trace_id" not in st.session_state:
    st.session_state.trace_id = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

# ── Query panel ───────────────────────────────────────────────────────────────
st.header("Query")
query_text = st.text_area("Enter your question", height=100, key="query_input")

if st.button("Submit", type="primary"):
    if not query_text.strip():
        st.warning("Query cannot be empty.")
    else:
        with st.spinner("Querying RAG pipeline…"):
            try:
                resp = requests.post(
                    f"{API_URL}/query/",
                    json={"query": query_text},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.trace_id = data.get("trace_id")
                    st.session_state.last_answer = data.get("answer")

                    st.success(data.get("answer", ""))

                    vcitations = data.get("vector_citations", [])
                    if vcitations:
                        with st.expander(f"Document citations ({len(vcitations)})"):
                            for c in vcitations:
                                st.markdown(
                                    f"**{c.get('doc_type')}** — {c.get('venue')} "
                                    f"({c.get('date')}) shift {c.get('shift_id')} "
                                    f"p.{c.get('page_num')} score={c.get('score', 0):.3f}"
                                )
                                st.caption(c.get("snippet", ""))

                    scitations = data.get("sql_citations", [])
                    if scitations:
                        with st.expander(f"SQL citations ({len(scitations)})"):
                            for c in scitations:
                                st.markdown(f"**View:** `{c.get('view_name')}`")
                                st.code(c.get("sql_used", ""), language="sql")
                                st.caption(f"{c.get('row_count')} rows returned")
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach API at {API_URL}. Is uvicorn running?")

# ── Feedback panel ────────────────────────────────────────────────────────────
if st.session_state.trace_id:
    st.divider()
    st.subheader("Was this answer helpful?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Yes"):
            requests.post(
                f"{API_URL}/feedback/",
                json={"trace_id": st.session_state.trace_id, "score": 1},
                timeout=5,
            )
            st.toast("Thanks for the feedback!")
    with col2:
        if st.button("👎 No"):
            requests.post(
                f"{API_URL}/feedback/",
                json={"trace_id": st.session_state.trace_id, "score": -1},
                timeout=5,
            )
            st.toast("Thanks — we'll use this to improve.")

# ── Ingestion panel ───────────────────────────────────────────────────────────
st.divider()
st.header("Ingest Document")
with st.form("ingest_form"):
    file_path = st.text_input("File path (server-side)", placeholder="/data/docs/procedure.pdf")
    col1, col2 = st.columns(2)
    with col1:
        doc_type = st.selectbox("Document type", ["procedure", "shift_log", "incident", "policy"])
        venue = st.text_input("Venue", placeholder="Fairfield RSL")
    with col2:
        date = st.text_input("Date (YYYY-MM-DD)", placeholder="2026-06-08")
        shift_id = st.text_input("Shift ID", placeholder="SH-001")
    submitted = st.form_submit_button("Ingest")

if submitted:
    if not file_path.strip():
        st.warning("File path is required.")
    else:
        with st.spinner("Ingesting document…"):
            try:
                resp = requests.post(
                    f"{API_URL}/ingest/sync",
                    json={
                        "file_path": file_path,
                        "doc_type": doc_type,
                        "venue": venue,
                        "date": date,
                        "shift_id": shift_id,
                    },
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"Ingested {data['chunks_ingested']} chunks "
                        f"({data['chunks_deduplicated']} deduplicated) — status: {data['status']}"
                    )
                else:
                    st.error(f"Ingest error {resp.status_code}: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach API at {API_URL}.")
