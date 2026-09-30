from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.ingestion.corpus_manifest import CorpusManifest, load_corpus_manifest

MANIFEST = Path("corpus/v1.yaml")


def test_checked_in_manifest_loads_with_stable_digest():
    first = load_corpus_manifest(MANIFEST)
    second = load_corpus_manifest(MANIFEST)

    assert first.corpus_version == "v1"
    assert first.document_collection == "rag_operational_v1"
    assert first.schema_collection == "rag_schema_meta_v1"
    assert first.embedding.dimension == 384
    assert len(first.documents) == 3
    assert first.digest == second.digest


def test_collection_names_must_match_version():
    data = yaml.safe_load(MANIFEST.read_text())
    data["document_collection"] = "rag_operational_v2"

    with pytest.raises(ValidationError, match="must end with the corpus version"):
        CorpusManifest.model_validate(data)


def test_duplicate_source_id_is_rejected():
    data = yaml.safe_load(MANIFEST.read_text())
    data["documents"][1]["source_id"] = data["documents"][0]["source_id"]

    with pytest.raises(ValidationError, match="duplicate document source_id"):
        CorpusManifest.model_validate(data)


def test_unknown_field_and_schema_version_are_rejected():
    data = yaml.safe_load(MANIFEST.read_text())
    data["unknown_setting"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CorpusManifest.model_validate(data)

    data.pop("unknown_setting")
    data["manifest_schema"] = 2
    with pytest.raises(ValidationError, match="Input should be 1"):
        CorpusManifest.model_validate(data)


def test_checksum_mismatch_is_rejected(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    source_dir = tmp_path / "tests" / "fixtures"
    source_dir.mkdir(parents=True)
    source = source_dir / "doc.txt"
    source.write_text("different content")
    data = yaml.safe_load(MANIFEST.read_text())
    data["documents"] = [{
        "source_id": "doc", "path": "tests/fixtures/doc.txt", "sha256": "0" * 64,
        "doc_type": "shift_log", "venue": "v", "date": "2026-01-01", "shift_id": "s",
    }]
    path = corpus_dir / "bad.yaml"
    path.write_text(yaml.safe_dump(data))

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_corpus_manifest(path)
