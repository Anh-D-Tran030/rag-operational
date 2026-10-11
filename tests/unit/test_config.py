from src.config import Settings


def test_v1_collection_defaults_are_versioned():
    config = Settings(_env_file=None)

    assert config.qdrant_collection == "rag_operational_v1"
    assert config.qdrant_schema_collection == "rag_schema_meta_v1"
    assert config.qdrant_api_key == ""


def test_active_collections_and_managed_qdrant_are_configurable(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "https://managed-qdrant.example")
    monkeypatch.setenv("QDRANT_API_KEY", "test-credential")
    monkeypatch.setenv("QDRANT_COLLECTION", "rag_operational_v2")
    monkeypatch.setenv("QDRANT_SCHEMA_COLLECTION", "rag_schema_meta_v2")

    config = Settings(_env_file=None)

    assert config.qdrant_url == "https://managed-qdrant.example"
    assert config.qdrant_api_key == "test-credential"
    assert config.qdrant_collection == "rag_operational_v2"
    assert config.qdrant_schema_collection == "rag_schema_meta_v2"
