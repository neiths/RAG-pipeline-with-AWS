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
    
    # pipecone configuration
    pinecone_api_key: str = Field(..., env="PIPECONE_API_KEY")
    pinecone_environment: str = Field(..., env="PINECONE_ENVIRONMENT")
    pinecone_index_name: Optional[str] = Field(default="rag-demo-index", env="PINECONE_INDEX_NAME")
    
    # # Text processing configuration
    chunk_size: int = Field(
        default=800,
        env="TEXT_CHUNK_SIZE",
        description="The size of text chunks to split documents into for processing.",
    )
    
    chunk_overlap: int = Field(
        default=100,
        env="TEXT_CHUNK_OVERLAP",
        description="The overlap size between text chunks.",
    )
    
    
    
    class Config: 
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"
    
    
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
    
def get_pipecone_config() -> dict:
    
    settings = get_settings()
    return {
        "api_key": settings.pinecone_api_key,
        "environment": settings.pinecone_environment,
        "index_name": settings.pinecone_index_name,
    }
    