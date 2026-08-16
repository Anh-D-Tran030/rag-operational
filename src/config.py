from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_operational"
    qdrant_schema_collection: str = "rag_schema_meta"
    ollama_url: str = "http://localhost:11434"
    ollama_model_router: str = "llama3.2:3b"
    # Routing, query expansion and simple-query generation. "ollama" keeps these on
    # a local GPU for development; "openrouter" is what lets the deployed container
    # run on plain CPU.
    local_provider: str = "openrouter"
    local_model: str = "mistralai/mistral-nemo"
    # Routing gets its own model: a misroute yields a confident answer from the wrong
    # source, which is worse than no answer. Few-shot prompting took this model to
    # 23/23 on the golden categories where local_model reaches 21/23. Query expansion
    # and simple generation stay on the cheaper local_model — they degrade gracefully.
    router_model: str = "qwen/qwen3-30b-a3b-instruct-2507"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Reasoning-grade but cheap: SQL generation and final answers.
    cloud_model: str = "openai/gpt-oss-120b"
    # High-volume, low-stakes: per-chunk contextual prefixes at ingest.
    context_model: str = "mistralai/mistral-nemo"
    # Faithfulness gate. Chosen by measurement, not price: on a grounded/ungrounded
    # probe, mistral-nemo passed 1/5 grounded answers and gpt-oss-120b passed 0/5,
    # while this model scored 5/5 both ways. A gate that rejects correct answers is
    # worse than no gate.
    gate_model: str = "qwen/qwen3-30b-a3b-instruct-2507"
    # RAGAS judge. This one is the measuring instrument — raising it costs little
    # and directly determines whether the published eval numbers mean anything.
    judge_model: str = "openai/gpt-oss-120b"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_fallback: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 50
    rerank_top_n: int = 7
    rerank_score_threshold: float = 0.3
    max_retrieval_iterations: int = 5
    token_budget_per_query: int = 4000
    db_url: str = "sqlite+aiosqlite:///./operational.db"
    semantic_cache_url: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = ""
    prometheus_port: int = 8001

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def local_llm(self, model: str | None = None):
        """Bundle the small-model settings for src.llm_client.complete.

        `model` overrides the default for callers that need a specific one, such
        as the router. Ignored under the ollama provider, which serves one model.
        """
        # Imported here rather than at module scope so config stays importable
        # from anywhere without dragging the SDK clients in.
        from src.llm_client import LocalLLMConfig

        return LocalLLMConfig(
            provider=self.local_provider,
            model=(
                self.ollama_model_router
                if self.local_provider == "ollama"
                else (model or self.local_model)
            ),
            ollama_url=self.ollama_url,
            api_key=self.openrouter_api_key,
            base_url=self.openrouter_base_url,
        )


settings = Settings()
