import datetime
from typing import Any, Dict, List
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from dotenv import load_dotenv

import os
import json

import logging

import hashlib
import pathlib

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

from config import get_settings

logger = logging.getLogger(__name__)

class DocumentIngestor:
    
    def __init__(self):
        """Initialize ingestor with AWS and pipecone clients"""
        self.settings = get_settings()
        self._setup_clients()
        self._setup_text_splitter()
           
    def _setup_clients(self):
        try: 
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=self.settings.aws_region,
            )
            
            logger.info("Bedrock client initialized successfully.")
            
            self._test_bedrock_connection()
            
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure AWS CLI or set environment variables.")
            raise
        except Exception as e:
            logger.error("Failed to initialize clients", error=str(e))
            raise
    
    def _test_bedrock_connection(self):
        try:
            
            test_response = self.bedrock_client.invoke_model(
                modelId=self.settings.embedding_model_id,
                body=json.dumps({"inputText": "test"}),
                contentType="application/json",
                accept="application/json",
            )
            logger.info("Bedrock connection test successful")
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'AccessDeniedException':
                logger.error(
                    "Access denied to Bedrock. Ensure the model is enabled in your region",
                    model_id=self.settings.embedding_model_id,
                    region=self.settings.aws_region
                )
            raise
    
    def _setup_text_splitter(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len   
        )
    
    def generate_embeddings(self, text: str) -> List[float]:
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.settings.embedding_model_id,
                body=json.dumps({"inputText": text}),
                contentType="application/json",
                accept="application/json",
            )
            
            response_body = json.loads(response['body'].read())
            embedding = response_body.get('embedding')
            
            return embedding
        
        except Exception as e:
            raise
    
    def create_document_id(self, source: str, chunk_index: int) -> str:
        content = f"{source}:{chunk_index}"
        
        return hashlib.md5(content.encode('utf-8')).hexdigest()
        
    def process_document(self, file_path: pathlib.Path) -> List[Dict[str, Any]]:
        try:
            loader = TextLoader(str(file_path), encoding='utf-8')
            
            documents = loader.load()
            
            if not documents:
                return []
            
            chunks = self.text_splitter.split_documents(documents)
            processed_chunks = []
            
            for i, chunk in enumerate(chunks):
                
                try:
                    embedding = self.generate_embeddings(chunk.page_content)
                    doc_id = self.create_document_id(str(file_path), i)
                    
                    metadata = {
                        "text": chunk.page_content,
                        "source": str(file_path),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "file_name": file_path.name,
                        "file_size": file_path.stat().st_size,
                        "ingested_at": datetime.datetime.now().isoformat(),
                        "text_lenght": len(chunk.page_content),
                    }
                    
                    processed_chunks.append({
                        "id": doc_id,
                        "embedding": embedding,
                        "metadata": metadata
                    })
                    
                except Exception as e:
                    continue
                
            return processed_chunks
        
        except Exception as e:
            return []

    def ingest_to_pinecone(self, text: str):
        pass
    
    def ingest_directory(self, directory: str):
        pass
    
    def ingest_s3_bucket(self, bucket_name: str, prefix: str = ""):
        pass
    
    def _serialize_document(self, text: str):
        pass
    

def main():
    
    load_dotenv()
    
    doc_ingestor = DocumentIngestor()
    
    proccessed_docs = doc_ingestor.process_document(
        pathlib.Path("test_data.txt")
    )
    
    print(f"Processed \n{proccessed_docs}\n chunks from the document.")

if __name__ == "__main__":
    main()