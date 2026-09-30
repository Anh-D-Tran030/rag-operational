"""Validated, human-readable definition of one deployable corpus version."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmbeddingSpec(ManifestModel):
    model: str
    dimension: int = Field(gt=0)


class ChunkingSpec(ManifestModel):
    chunk_size: int = Field(gt=0)
    overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> ChunkingSpec:
        if self.overlap >= self.chunk_size:
            raise ValueError("chunking.overlap must be smaller than chunk_size")
        return self


class DocumentSource(ManifestModel):
    source_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    doc_type: str = Field(min_length=1)
    venue: str
    date: str
    shift_id: str

    def metadata(self) -> dict[str, str]:
        return self.model_dump(include={"doc_type", "venue", "date", "shift_id"})


class StructuredSource(ManifestModel):
    table: Literal["shift_records", "incident_logs", "staffing_history"]
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SchemaView(ManifestModel):
    view_name: str = Field(min_length=1)
    columns: list[str] = Field(min_length=1)
    description: str = Field(min_length=1)
    example_query: str = Field(pattern=r"(?i)^SELECT\s")


class CorpusManifest(ManifestModel):
    manifest_schema: Literal[1] = 1
    corpus_version: str = Field(pattern=r"^v[1-9][0-9]*$")
    document_collection: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    schema_collection: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    embedding: EmbeddingSpec
    contextualization_model: str = Field(min_length=1)
    chunking: ChunkingSpec
    documents: list[DocumentSource] = Field(min_length=1)
    structured_data: list[StructuredSource] = Field(min_length=1)
    schema_views: list[SchemaView] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identity(self) -> CorpusManifest:
        suffix = f"_{self.corpus_version}"
        if not self.document_collection.endswith(suffix):
            raise ValueError("document_collection must end with the corpus version")
        if not self.schema_collection.endswith(suffix):
            raise ValueError("schema_collection must end with the corpus version")
        if self.document_collection == self.schema_collection:
            raise ValueError("document and schema collections must differ")
        self._require_unique("document source_id", [item.source_id for item in self.documents])
        self._require_unique("document path", [item.path for item in self.documents])
        self._require_unique("structured table", [item.table for item in self.structured_data])
        self._require_unique("schema view", [item.view_name for item in self.schema_views])
        return self

    @staticmethod
    def _require_unique(label: str, values: list[str]) -> None:
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {label} in manifest")

    @property
    def digest(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def _resolve_source(project_root: Path, relative_path: str) -> Path:
    source = (project_root / relative_path).resolve()
    if not source.is_relative_to(project_root):
        raise ValueError(f"source path escapes project root: {relative_path}")
    return source


def load_corpus_manifest(path: str | Path, *, verify_files: bool = True) -> CorpusManifest:
    manifest_path = Path(path).resolve()
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest = CorpusManifest.model_validate(data)
    if not verify_files:
        return manifest

    project_root = manifest_path.parent.parent
    sources = [*manifest.documents, *manifest.structured_data]
    for item in sources:
        source = _resolve_source(project_root, item.path)
        if not source.is_file():
            raise ValueError(f"manifest source does not exist: {item.path}")
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != item.sha256:
            raise ValueError(f"checksum mismatch for manifest source: {item.path}")
    return manifest


def resolve_manifest_source(manifest_path: str | Path, relative_path: str) -> Path:
    """Resolve a manifest source relative to the repository root."""
    path = Path(manifest_path).resolve()
    return _resolve_source(path.parent.parent, relative_path)
