from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://cortex:cortex@localhost:5432/cortex"

    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    embed_batch_size: int = 64

    # Token estimates keep passage inputs under bge-small-en-v1.5's 512 token cap.
    chunk_target_tokens: int = 350
    chunk_max_tokens: int = 480
    chunk_overlap_tokens: int = 40

    # Layer 3 query-side use only. Do not prepend this to passages in Layer 2.
    bge_query_instruction: str = "Represent this sentence for searching relevant passages:"

    # Layer 3: chat (RAG).
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_disable_thinking: bool = True
    chat_temperature: float = 0.2
    chat_max_tokens: int = 800
    chat_timeout_s: float = 60.0
    chat_max_retries: int = 2

    retrieval_k: int = 6
    retrieval_candidates: int = 20
    retrieval_hybrid: bool = True
    rrf_candidates: int = 50
    rrf_k: int = 60
    chat_context_char_cap: int = 6000


settings = Settings()
