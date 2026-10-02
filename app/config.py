from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 这三项故意不给默认值,缺了在启动就报 Field required,直接指向 .env
    chat_model: str
    chat_base_url: str
    chat_api_key: str
    token_budget: int = 2000
    model_context_window: int = Field(default=1_000_000,gt=0)
    max_output_tokens: int = Field(default=4000,gt=0)
    max_user_input_tokens: int = Field(default=4000,gt=0)
    max_agent_steps: int = Field(default=5,gt=0)
    tool_result_max_tokens: int = Field(default=2400,gt=0)
    context_desired_turns: int = Field(default=40,gt=0)
    agent_max_tool_steps: int = 5
    agent_max_model_calls: int = 6
    agent_max_tokens: int = 12000
    agent_timeout_seconds: float = 120
    workflow_gate_threshold: float = 0.032755612017354
    workflow_checkpoint_path: str = "data/runtime/ch05-checkpoints.sqlite"
    intent_small_model: str = ""
    intent_confidence_threshold: float = Field(default=0.75,gt=0,le=1,allow_inf_nan=False)
    database_url: str = "mysql+pymysql://mewhelp:mewhelp_local@127.0.0.1:3307/mewhelp?charset=utf8mb4"
    embed_base_url: str = "https://api.siliconflow.cn/v1"
    embed_model: str = "BAAI/bge-m3"
    embed_api_key: str = ""
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = "knowledge_ch04"
    recall_top_k: int = 50
    rerank_base_url: str = "https://api.siliconflow.cn/v1"
    rerank_api_key: str = ""
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_k: int = 10
    rerank_min_score: float = 0.0
    rag_top_k: int = 5
    rag_min_score: float = 0.55
    # Exact Milvus category values; empty disables category filtering.
    rag_category_allowlist: tuple[str, ...] = ()

settings = Settings()
