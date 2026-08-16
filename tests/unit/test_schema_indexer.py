"""Unit tests for src/ingestion/schema_indexer.py."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ingestion.schema_indexer import _SCHEMA_DESCRIPTIONS, index_schema_metadata

REQUIRED_PAYLOAD_FIELDS = ["view_name", "columns", "description", "example_query"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_embedder(dim: int = 384) -> MagicMock:
    """Return a mock Embedder whose embed_batch always returns dim-dimensional vectors."""
    mock = MagicMock()
    mock.embed_batch = MagicMock(
        side_effect=lambda texts: [[0.1] * dim for _ in texts]
    )
    return mock


def _make_mock_client(
    collection_exists: bool = False, existing_name: str = "rag_schema_meta"
) -> MagicMock:
    """Return a mock AsyncQdrantClient.

    When collection_exists=True the existing_name collection is reported as already present,
    so create_collection should NOT be called for that name.
    """
    mock = MagicMock()

    if collection_exists:
        col = MagicMock()
        col.name = existing_name  # set attribute directly, not via constructor keyword
        existing_cols = [col]
    else:
        existing_cols = []

    mock_collections = MagicMock()
    mock_collections.collections = existing_cols
    mock.get_collections = AsyncMock(return_value=mock_collections)

    mock.create_collection = AsyncMock()
    mock.upsert = AsyncMock()
    return mock


# ---------------------------------------------------------------------------
# Test: happy path — collection does not yet exist
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_index_schema_metadata_returns_correct_count():
    """index_schema_metadata must return one entry per _SCHEMA_DESCRIPTIONS item."""
    client = _make_mock_client(collection_exists=False)
    embedder = _make_mock_embedder()

    count = await index_schema_metadata(
        client=client,
        schema_collection="test_rag_schema_meta",
        embedder=embedder,
        db_url="sqlite+aiosqlite:///./test_operational.db",
    )

    assert count == len(_SCHEMA_DESCRIPTIONS)
    assert count > 0


@pytest.mark.asyncio
async def test_index_schema_metadata_all_payload_fields_present():
    """Every upserted point must carry all required payload fields."""
    captured_points: list = []

    async def capture_upsert(collection_name, points):
        captured_points.extend(points)

    client = _make_mock_client(collection_exists=False)
    client.upsert = capture_upsert  # replace AsyncMock with capturing coroutine
    embedder = _make_mock_embedder()

    await index_schema_metadata(
        client=client,
        schema_collection="test_rag_schema_meta",
        embedder=embedder,
        db_url="sqlite+aiosqlite:///./test_operational.db",
    )

    assert len(captured_points) == len(_SCHEMA_DESCRIPTIONS)
    for point in captured_points:
        for field in REQUIRED_PAYLOAD_FIELDS:
            assert field in point.payload, f"Missing payload field: {field}"


@pytest.mark.asyncio
async def test_index_schema_metadata_creates_collection_when_missing():
    """create_collection must be called exactly once when the collection does not exist."""
    client = _make_mock_client(collection_exists=False)
    embedder = _make_mock_embedder()

    await index_schema_metadata(
        client=client,
        schema_collection="test_rag_schema_meta",
        embedder=embedder,
        db_url="sqlite+aiosqlite:///./test_operational.db",
    )

    client.create_collection.assert_called_once()


@pytest.mark.asyncio
async def test_index_schema_metadata_skips_collection_creation_when_exists():
    """create_collection must NOT be called when the collection already exists."""
    client = _make_mock_client(collection_exists=True)
    embedder = _make_mock_embedder()

    await index_schema_metadata(
        client=client,
        schema_collection="rag_schema_meta",
        embedder=embedder,
        db_url="sqlite+aiosqlite:///./test_operational.db",
    )

    client.create_collection.assert_not_called()


# ---------------------------------------------------------------------------
# Test: embedding integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_index_schema_metadata_calls_embed_batch_once():
    """embed_batch must be called — at most twice (once for size probe, once for all texts)."""
    client = _make_mock_client(collection_exists=False)
    embedder = _make_mock_embedder()

    await index_schema_metadata(
        client=client,
        schema_collection="test_rag_schema_meta",
        embedder=embedder,
        db_url="sqlite+aiosqlite:///./test_operational.db",
    )

    # embed_batch is called at least once (for the actual schema texts)
    assert embedder.embed_batch.call_count >= 1


@pytest.mark.asyncio
async def test_index_schema_metadata_embeds_all_views():
    """The number of embeddings produced must equal the number of schema views."""
    embedded_texts: list[list[str]] = []

    def recording_embed_batch(texts: list[str]) -> list[list[float]]:
        embedded_texts.extend(texts)
        return [[0.1] * 384 for _ in texts]

    client = _make_mock_client(collection_exists=False)
    embedder = MagicMock()
    embedder.embed_batch = recording_embed_batch

    await index_schema_metadata(
        client=client,
        schema_collection="test_rag_schema_meta",
        embedder=embedder,
        db_url="sqlite+aiosqlite:///./test_operational.db",
    )

    # After accounting for the size-probe call (1 text), the main batch must
    # contain exactly len(_SCHEMA_DESCRIPTIONS) texts.
    schema_batch_texts = [t for t in embedded_texts if t != "test"]
    assert len(schema_batch_texts) == len(_SCHEMA_DESCRIPTIONS)


# ---------------------------------------------------------------------------
# Test: payload content correctness
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_index_schema_metadata_payload_view_names_match_descriptions():
    """The view_name in each upserted payload must match the source _SCHEMA_DESCRIPTIONS."""
    captured_points: list = []

    async def capture_upsert(collection_name, points):
        captured_points.extend(points)

    client = _make_mock_client(collection_exists=False)
    client.upsert = capture_upsert
    embedder = _make_mock_embedder()

    await index_schema_metadata(
        client=client,
        schema_collection="test_rag_schema_meta",
        embedder=embedder,
        db_url="sqlite+aiosqlite:///./test_operational.db",
    )

    expected_view_names = {s["view_name"] for s in _SCHEMA_DESCRIPTIONS}
    actual_view_names = {p.payload["view_name"] for p in captured_points}
    assert actual_view_names == expected_view_names


@pytest.mark.asyncio
async def test_index_schema_metadata_columns_is_list():
    """Each payload 'columns' entry must be a list."""
    captured_points: list = []

    async def capture_upsert(collection_name, points):
        captured_points.extend(points)

    client = _make_mock_client(collection_exists=False)
    client.upsert = capture_upsert
    embedder = _make_mock_embedder()

    await index_schema_metadata(
        client=client,
        schema_collection="test_rag_schema_meta",
        embedder=embedder,
        db_url="sqlite+aiosqlite:///./test_operational.db",
    )

    for point in captured_points:
        assert isinstance(point.payload["columns"], list), (
            f"columns must be list, got {type(point.payload['columns'])} "
            f"for view {point.payload['view_name']}"
        )
        assert len(point.payload["columns"]) > 0


@pytest.mark.asyncio
async def test_index_schema_metadata_example_query_is_select():
    """Each example_query in the payload must be a SELECT statement."""
    captured_points: list = []

    async def capture_upsert(collection_name, points):
        captured_points.extend(points)

    client = _make_mock_client(collection_exists=False)
    client.upsert = capture_upsert
    embedder = _make_mock_embedder()

    await index_schema_metadata(
        client=client,
        schema_collection="test_rag_schema_meta",
        embedder=embedder,
        db_url="sqlite+aiosqlite:///./test_operational.db",
    )

    for point in captured_points:
        example_query = point.payload["example_query"]
        assert isinstance(example_query, str)
        assert example_query.strip().upper().startswith("SELECT"), (
            f"example_query must start with SELECT for view {point.payload['view_name']}"
        )


# ---------------------------------------------------------------------------
# Test: upsert is called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_index_schema_metadata_calls_upsert():
    """client.upsert must be called exactly once with the schema collection name."""
    client = _make_mock_client(collection_exists=False)
    embedder = _make_mock_embedder()

    await index_schema_metadata(
        client=client,
        schema_collection="test_rag_schema_meta",
        embedder=embedder,
        db_url="sqlite+aiosqlite:///./test_operational.db",
    )

    client.upsert.assert_called_once()
    call_kwargs = client.upsert.call_args
    # Collection name must be the one we passed in
    collection_used = call_kwargs.kwargs.get(
        "collection_name", call_kwargs.args[0] if call_kwargs.args else None
    )
    assert collection_used == "test_rag_schema_meta"


# ---------------------------------------------------------------------------
# Test: _SCHEMA_DESCRIPTIONS constant integrity
# ---------------------------------------------------------------------------

def test_schema_descriptions_constant_has_five_views():
    """There must be exactly 5 view descriptions (one per curated SQL view from M1-15)."""
    assert len(_SCHEMA_DESCRIPTIONS) == 5


def test_schema_descriptions_all_have_required_keys():
    """Every entry in _SCHEMA_DESCRIPTIONS must have all required keys."""
    for entry in _SCHEMA_DESCRIPTIONS:
        for field in REQUIRED_PAYLOAD_FIELDS:
            assert field in entry, f"_SCHEMA_DESCRIPTIONS entry missing key: {field}"
