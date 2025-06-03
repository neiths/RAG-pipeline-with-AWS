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
    


    