import json
import logging
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from pinecone import Pinecone
import structlog

from .config import get_settings, Settings

logger = structlog.get_logger()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The query string to search for.")
    max_chunks: Optional[int] = Field(
        default=None, 
        ge=1, 
        le=20, 
        description="Maximum number of chunks to return. Defaults to 10 if not provided."
    )
    similarity_threshold: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0, 
        description="Minimum similarity score to return a chunk. Defaults to 0.5 if not provided."
    )
    include_sources: Optional[bool] = Field(default=True, description="Include source information in the response.")
    
    @field_validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace.")
        return v.strip()
    
class SourceInfo(BaseModel):
    source: str 
    chunk_index: int 
    similarity_score: float 
    text_preview: str = Field(description="First 200 characters of the chunk text.")
    
class QueryResponse(BaseModel):
    answer: str 
    query: str
    sources: List[SourceInfo] = Field(default_factory=list)
    processing_time_ms: float 
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class HealthResponse(BaseModel):
    status: str 
    timestamp: str
    version: str
    services: Dict[str, str]
    
class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pinecone = Pinecone(settings.pinecone_api_key, environment=settings.pinecone_environment)
        self.s3_client = boto3.client('s3', region_name=settings.aws_region)
        self.s3_bucket = settings.s3_bucket
        self.index_name = settings.pinecone_index_name
        
    async def query(self, request: QueryRequest) -> QueryResponse:
        # Implement the logic to query Pinecone and S3 here
        pass

    async def health_check(self) -> HealthResponse:
        # Implement health check logic here
        pass
    
    
def create_app() -> FastAPI:
    settings = get_settings()
    
    # Initialize FastAPI app
    app = FastAPI(
        title="Document Query Service", 
        description="A service to query documents using Pinecone and AWS.",
        version=settings.version,
        docs_url="/docs",
        redoc_url="/redoc",   
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app

app = create_app()

rag_service = None

@app.on_event("startup")
async def startup_event():
    global rag_service
    
    try:
        settings = get_settings()
        rag_service = RAGService(settings)
        logging.info("RAG Service initialized successfully.")
    except Exception as e:
        logger.error("failed to start application", error=str(e))
        raise

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application.")
    
    
    
if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.reload, 
        log_level=settings.log_level.lower()
    )