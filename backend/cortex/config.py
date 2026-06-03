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


settings = Settings()
