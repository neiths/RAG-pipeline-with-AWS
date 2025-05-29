import os 

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    
    # aws configuration
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    
    # bedrock configuration
    embedding_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        env="BEDROCK_EMEDDING_MODEL_ID",
        description="The model ID for the embedding model to use in Bedrock.",
    )
    
    # llm_model_id: str = Field(
    #     default="us.anthropic.claude-sonnet-4-20250514-v1:0",
    #     env="BEDROCK_LLM_MODEL_ID",
    #     description="The model ID for the LLM to use in Bedrock.",
    # )
    
    # # pipecone configuration
    # pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    # pinecone_env: str = Field(..., env="PINECONE_ENVIRONMENT")
    # pinecone_index_name: str = Field(default="rag-index", env="PINECONE_INDEX_NAME")
    
    # # Text processing configuration
    
@lru_cache()
def get_settings() -> Settings:
    return Settings()

def get_aws_region() -> str:
    return get_settings().aws_region

def get_bedrock_config() -> dict:
    settings = get_settings()
    return {
        "region": settings.aws_region,
        "embedding_model_id": settings.embedding_model_id,
    }
    