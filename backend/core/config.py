from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel,Field
from typing import List, Dict,Optional,Set
from functools import lru_cache


def normalize_storage(
    data: Dict[str, Optional[list[str]]]
) -> Dict[str, Optional[Set[str]]]:

    return {
        k: set(v) if v is not None else None
        for k, v in data.items()
    }

class CrossLanguageStrategyConfig(BaseModel):
    enabled: bool = True
    min_chunks_to_analyze: int = 5
    language_confidence_threshold: float = 0.7

class Config(BaseSettings):

    APP_NAME : str 
    APP_VERSION : str 

    DEBUG:bool 
    SUPERUSER_EMAIL:str
    SUPERUSER_USERNAME:str
    SUPERUSER_PASSWORD:str
# =====================================
    SUPPORTED_LANGUAGES:List[str]
    DEFAULT_LANGUAGE:str
    ALLOWED_EXTENSIONS:List[str]
    MAX_FILE_SIZE_BYTES:int
    CROSS_LANGUAGE_STRATEGY: CrossLanguageStrategyConfig = Field(
        default_factory=CrossLanguageStrategyConfig
    )
# =====================================
    SUPPORTED_CHUNKS:List[str]
    DEFAULT_CHUNK:str
    CHUNK_SIZE:int
    CHUNK_OVERLAP:int
    CHUNK_MIN_SIZE:int
# =====================================
    EMBED_DIM:int

    SUPPORTED_EMBED:List[str]
    DEFAULT_EMBED:str

    JINA_API_KEY:str
    JINA_BASE_URL:str
    JINA_MODEL:str

    MAX_TOKENS_LATE_CHUNKING:int   
    LATE_CHUNKING:bool
    BATCH_SIZE:int
    TIME_OUT:int
    MAX_RETRIES:int
    MAX_CONNECTIONS:int
# =====================================
    RERANKER_MODEL:str
    RERANKER_API_PROVIDER:str
# =====================================
    DEFAULT_VECTOR_STORE :str
    SUPPORTED_VECTORS: List[str]
# =====================================
    SUPPORTED_STORAGE_json: Dict[str, Optional[list[str]]]
    @property
    def SUPPORTED_STORAGE(self):
        return normalize_storage(self.SUPPORTED_STORAGE_json)    
    DEFAULT_STORAGE:str

    SUPPORTED_DB :List[str]
    DEFAULT_DB : str

    POSTGRES_USERNAME: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT : int
    POSTGRES_DATABASE_NAME: str

# =====================================
    QUERY_GROQ_API_KEY:str
    QUERY_GROQ_API_URL:str
    QUERY_MODEL:str
    QUERY_MAX_NEW_TOKENS:int
    QUERY_DEFAULT_TIMEOUT:float
# =====================================
    HISTORY_TURNS: int 
    GENERATOR_PROVIDERS :List[str]
    DEFAULT_GENERATOR_PROVIDER:str

    # Answer Generator
    GENERATOR_GROQ_API_KEY:str
    GENERATOR_GROQ_API_URL:str
    GENERATOR_GROQ_MODEL:str
    GENERATOR_GROQ_DAILY_LIMIT:int

    #  ـــــــــــــــHuggingFace API Settings (Backup)

    GENERATOR_HF_API_TOKEN:str
    GENERATOR_HF_MODEL:str

    # ـــــــــــــــContext & Generator Settings
    MAX_CONTEXT_CHARS:int
    MAX_CHUNK_CHARS:int
    GENERATOR_MAX_TOKENS:int
    GENERATOR_DEFAULT_TIMEOUT:float

    # ــــــــــــــ Smart Guard Thresholds
    GENERATOR_SOFT_THRESHOLD_PCT:float
    GENERATOR_HARD_THRESHOLD_PCT:float
    # ــــــــــــــ Evaluation
    SUPPORTED_EVALUATORS:List[str]
    DEFAULT_EVALUATOR:str
    COHERE_API_KEY :str
    COHERE_EVAL_MODEL:str

    # ــــــــــــــ Security
    SECRET_KEY:str
    JWT_ALGORITHM:str
    ACCESS_TOKEN_EXPIRE_MINUTES:int
    REFRESH_TOKEN_EXPIRE_DAYS:int

    # ــــــــــــــ 
    REDIS_URL:str
# =====================================
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        case_sensitive=True,
        extra="ignore",)
    


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config()
