"""Unit tests for src/retrieval/reranker.py."""

import subprocess
import sys

import pytest

from src.retrieval.reranker import _sync_rerank, rerank

CHUNKS = [
    {
        "content": "The Q1 sales report shows strong growth in the North region.",
        "content_hash": "a1",
    },
    {"content": "Employee handbook section on leave policy.", "content_hash": "a2"},
    {"content": "Q1 revenue increased by 15% compared to last year.", "content_hash": "a3"},
    {"content": "Office supply order submitted on Monday.", "content_hash": "a4"},
]


def test_reranker_executor_cleanup_allows_subprocess_to_exit() -> None:
    """The executor lifecycle must not keep a completed test process alive."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from src.retrieval.reranker import _get_executor, "
                "shutdown_reranker_executor; "
                "_get_executor().submit(lambda: None).result(); "
                "shutdown_reranker_executor(); print('exited')"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "exited"


@pytest.mark.asyncio
async def test_top_n_enforced():
    result = await rerank(
        query="Q1 sales",
        chunks=CHUNKS,
        top_n=2,
        score_threshold=0.0,
        trace_id="trace-001",
    )
    assert len(result) <= 2


@pytest.mark.asyncio
async def test_rerank_score_field_present():
    result = await rerank(
        query="Q1 sales revenue",
        chunks=CHUNKS,
        top_n=4,
        score_threshold=0.0,
        trace_id="trace-002",
    )
    for chunk in result:
        assert "rerank_score" in chunk
        assert isinstance(chunk["rerank_score"], float)


@pytest.mark.asyncio
async def test_hard_reject_below_threshold():
    # Use a threshold of 1.0 which no chunk should exceed
    result = await rerank(
        query="completely unrelated gibberish xyzzy",
        chunks=CHUNKS,
        top_n=4,
        score_threshold=1.0,
        trace_id="trace-003",
    )
    assert result == []


@pytest.mark.asyncio
async def test_trace_id_param_accepted():
    # Ensures trace_id is a named param (not relying on contextvars)
    result = await rerank(
        query="sales",
        chunks=CHUNKS[:2],
        top_n=2,
        score_threshold=0.0,
        trace_id="explicit-trace-id",
    )
    assert isinstance(result, list)


class _StubRanker:
    def rerank(self, request):
        return [
            {"id": 0, "score": 0.11},
            {"id": 1, "score": 0.10},
            {"id": 2, "score": 0.09},
        ]


def test_threshold_retains_candidates_at_or_above_point_one(monkeypatch) -> None:
    monkeypatch.setattr("src.retrieval.reranker._get_ranker", lambda: _StubRanker())

    result = _sync_rerank(
        query="irrelevant for stub",
        chunks=CHUNKS[:3],
        top_n=7,
        score_threshold=0.1,
        trace_id="threshold-regression",
    )

    assert [chunk["content_hash"] for chunk in result] == ["a1", "a2"]
    assert [chunk["rerank_score"] for chunk in result] == [0.11, 0.10]


class _CapturingRanker:
    passages: list[dict] = []

    def rerank(self, request):
        self.passages = request.passages
        return [{"id": 1, "score": 0.9}, {"id": 0, "score": 0.5}]


def test_passage_uses_prefix_when_present_and_maps_to_original(monkeypatch) -> None:
    ranker = _CapturingRanker()
    monkeypatch.setattr("src.retrieval.reranker._get_ranker", lambda: ranker)
    chunks = [
        {"content": "Two signatures.", "contextual_prefix": "SOP 2.3 approvals.", "id": "c0"},
        {"content": "Leave policy.", "contextual_prefix": "", "id": "c1"},
    ]

    result = _sync_rerank("q", chunks, top_n=7, score_threshold=0.1, trace_id="t")

    texts = ["SOP 2.3 approvals. Two signatures.", "Leave policy."]
    assert [p["text"] for p in ranker.passages] == texts
    assert [c["id"] for c in result] == ["c1", "c0"]
    assert result[1] == {**chunks[0], "rerank_score": 0.5}
    assert "rerank_score" not in chunks[0]


def test_passage_is_content_alone_when_prefix_missing(monkeypatch) -> None:
    ranker = _CapturingRanker()
    monkeypatch.setattr("src.retrieval.reranker._get_ranker", lambda: ranker)

    _sync_rerank("q", CHUNKS[:2], top_n=7, score_threshold=0.1, trace_id="t")

    assert [p["text"] for p in ranker.passages] == [c["content"] for c in CHUNKS[:2]]


# case 13: narrow sign-off selection (contract item C).
_SIGN_OFF_QUERY = "Who signed off the shift log on 2026-05-28?"

_CASH_RECON = {
    "content": "Cash reconciliation completed by J. Thompson.",
    "doc_type": "shift_log", "date": "2026-05-28", "chunk_id": "shift_log.txt:patron_count",
}
_SUPERVISOR_SIGN_OFF = {
    "content": "Supervisor sign-off: J. Thompson — 2026-05-28 00:32",
    "doc_type": "shift_log", "date": "2026-05-28", "chunk_id": "shift_log.txt:sign_off",
}
_SIGNED_OFF_BY = {
    "content": "Shift handover notes. Signed off by M. Nguyen.",
    "doc_type": "shift_log", "date": "2026-05-28", "chunk_id": "shift_log.txt:handover",
}
_WRONG_DATE_SIGN_OFF = {
    "content": "Supervisor sign-off: J. Thompson — 2026-05-27 23:58",
    "doc_type": "shift_log", "date": "2026-05-27", "chunk_id": "shift_log.txt:sign_off_prev",
}
_UNRELATED_HEADER_WITH_SAME_NAME = {
    "content": "Roster: supervisor on duty was J. Thompson.",
    "doc_type": "shift_log", "date": "2026-05-28", "chunk_id": "shift_log.txt:roster",
}


class _RankerReturning:
    """Stub ranker returning fixed (id, score) pairs, in given input order."""

    def __init__(self, scores: list[float]):
        self._scores = scores

    def rerank(self, request):
        return [{"id": i, "score": s} for i, s in enumerate(self._scores)]


def test_cash_reconciliation_scores_higher_but_true_sign_off_is_selected(monkeypatch):
    chunks = [_CASH_RECON, _SUPERVISOR_SIGN_OFF]
    monkeypatch.setattr(
        "src.retrieval.reranker._get_ranker",
        lambda: _RankerReturning([0.997579, 0.994263]),  # cash recon scores higher
    )
    result = _sync_rerank(_SIGN_OFF_QUERY, chunks, top_n=7, score_threshold=0.0, trace_id="t")
    assert [c["chunk_id"] for c in result] == ["shift_log.txt:sign_off"]


def test_same_name_in_unrelated_header_is_not_sign_off_evidence(monkeypatch):
    chunks = [_UNRELATED_HEADER_WITH_SAME_NAME, _SUPERVISOR_SIGN_OFF]
    monkeypatch.setattr(
        "src.retrieval.reranker._get_ranker", lambda: _RankerReturning([0.9, 0.8])
    )
    result = _sync_rerank(_SIGN_OFF_QUERY, chunks, top_n=7, score_threshold=0.0, trace_id="t")
    assert [c["chunk_id"] for c in result] == ["shift_log.txt:sign_off"]


def test_wrong_date_sign_off_is_excluded(monkeypatch):
    chunks = [_WRONG_DATE_SIGN_OFF, _CASH_RECON]
    monkeypatch.setattr(
        "src.retrieval.reranker._get_ranker", lambda: _RankerReturning([0.99, 0.5])
    )
    # Neither candidate qualifies for 2026-05-28: wrong date, and no sign-off
    # evidence in the other. Falls back to the unchanged FlashRank list.
    result = _sync_rerank(_SIGN_OFF_QUERY, chunks, top_n=7, score_threshold=0.0, trace_id="t")
    assert [c["chunk_id"] for c in result] == ["shift_log.txt:sign_off_prev", "shift_log.txt:patron_count"]


def test_multiple_sign_offs_are_ranked_by_existing_score(monkeypatch):
    other_signer = {
        "content": "Supervisor sign-off: M. Nguyen — 2026-05-28 07:00",
        "doc_type": "shift_log", "date": "2026-05-28", "chunk_id": "shift_log.txt:sign_off_2",
    }
    chunks = [other_signer, _SUPERVISOR_SIGN_OFF]
    monkeypatch.setattr(
        "src.retrieval.reranker._get_ranker", lambda: _RankerReturning([0.6, 0.9])
    )
    result = _sync_rerank(_SIGN_OFF_QUERY, chunks, top_n=7, score_threshold=0.0, trace_id="t")
    assert [c["chunk_id"] for c in result] == ["shift_log.txt:sign_off", "shift_log.txt:sign_off_2"]


def test_missing_sign_off_falls_back_to_unchanged_flashrank_list(monkeypatch):
    chunks = [_CASH_RECON, _UNRELATED_HEADER_WITH_SAME_NAME]
    monkeypatch.setattr(
        "src.retrieval.reranker._get_ranker", lambda: _RankerReturning([0.99, 0.5])
    )
    result = _sync_rerank(_SIGN_OFF_QUERY, chunks, top_n=7, score_threshold=0.0, trace_id="t")
    assert [c["chunk_id"] for c in result] == ["shift_log.txt:patron_count", "shift_log.txt:roster"]


def test_non_sign_off_query_is_unaffected_by_narrowing(monkeypatch):
    chunks = [_CASH_RECON, _SUPERVISOR_SIGN_OFF]
    monkeypatch.setattr(
        "src.retrieval.reranker._get_ranker", lambda: _RankerReturning([0.99, 0.5])
    )
    result = _sync_rerank(
        "What is the peak patron count?", chunks, top_n=7, score_threshold=0.0, trace_id="t"
    )
    assert [c["chunk_id"] for c in result] == ["shift_log.txt:patron_count", "shift_log.txt:sign_off"]


def test_sign_off_narrowing_does_not_depend_on_patron_count_chunk_id(monkeypatch):
    renamed = dict(_SUPERVISOR_SIGN_OFF, chunk_id="anything:else")
    chunks = [_CASH_RECON, renamed]
    monkeypatch.setattr(
        "src.retrieval.reranker._get_ranker", lambda: _RankerReturning([0.99, 0.94])
    )
    result = _sync_rerank(_SIGN_OFF_QUERY, chunks, top_n=7, score_threshold=0.0, trace_id="t")
    assert [c["chunk_id"] for c in result] == ["anything:else"]


def test_sign_off_narrowing_respects_top_n_and_threshold(monkeypatch):
    chunks = [_SUPERVISOR_SIGN_OFF, _CASH_RECON]
    monkeypatch.setattr(
        "src.retrieval.reranker._get_ranker", lambda: _RankerReturning([0.994263, 0.01])
    )
    # score_threshold excludes the cash-recon candidate before narrowing even runs.
    result = _sync_rerank(_SIGN_OFF_QUERY, chunks, top_n=7, score_threshold=0.5, trace_id="t")
    assert [c["chunk_id"] for c in result] == ["shift_log.txt:sign_off"]
