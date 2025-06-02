import datetime
from typing import Any, Dict, List
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from dotenv import load_dotenv

import os
import json

import logging

from datetime import datetime, UTC

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
            
            self.pc = Pinecone(
                api_key=self.settings.pinecone_api_key,
                environment=self.settings.pinecone_environment,
            )
            
            # if self.pc.has_index(self.settings.pinecone_index_name):
            #     self.pc.create_index_for_model(
            #             name=self.settings.pinecone_index_name,
            #             cloud="aws",
            #             region=self.settings.aws_region,
            #             embed={
            #                 "model":"llama-text-embed-v2",
            #                 "field_map":{"text": "chunk_text"}
            #             }
            #         )
            
            self.pinecone_index = self.pc.Index(
                self.settings.pinecone_index_name,
            )
            
            # logger.info("Pinecone client initialized", index=self.settings.pinecone_index_name)
            
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

    def ingest_to_pinecone(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        
        if not chunks:
            logger.warning("No chunks to ingest")
            return
        
        logger.info("Starting Pinecone ingestion", total_chunks=len(chunks))
        
        try:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                vectors = [
                    (chunk['id'], chunk['values'], chunk['metadata']) 
                    for chunk in batch
                ]
                
                self.pinecone_index.upsert(
                    vectors=vectors,
                )
                
                logger.info(
                    "Batch uploaded to Pinecone",
                    batch_start=i,
                    batch_size=len(batch),
                    total_uploaded=min(i + batch_size, len(chunks))
                )
                
            logger.info("Pinecone ingestion complete", total_chunks=len(chunks))
            
        except Exception as e:
            logger.error("Failed to ingest to Pinecone", error=str(e))
            raise
    
    def ingest_directory(self, directory_path: str):

        directory = pathlib.Path(directory_path)
        
        if not directory.exists():
            logger.error("Directory does not exist", directory=str(directory))
            return
        
        # Find all text files
        text_files = list(directory.rglob("*.txt"))
        
        if not text_files:
            logger.warning("No .txt files found in directory", directory=str(directory))
            return
        
        logger.info("Starting directory ingestion", directory=str(directory), file_count=len(text_files))
        
        all_chunks = []
        successful_files = 0
        
        for file_path in text_files:
            try:
                chunks = self.process_document(file_path)
                all_chunks.extend(chunks)
                successful_files += 1
                
            except Exception as e:
                logger.error("Failed to process file", file_path=str(file_path), error=str(e))
                continue
        
        if all_chunks:
            self.ingest_to_pinecone(all_chunks)
            
            # Print summary
            try:
                index_stats = self.pinecone_index.describe_index_stats()
                # index_stats_dict = self._serialize_index_stats(index_stats)
            except Exception as e:
                logger.warning("Failed to get index stats", error=str(e))
                # index_stats_dict = {"error": "Unable to retrieve stats"}
            
            try:
                logger.info(
                    "Ingestion complete",
                    files_processed=successful_files,
                    total_files=len(text_files),
                    chunks_ingested=len(all_chunks),
                    index_stats=None
                )
            except Exception as e:
                import traceback
                logger.error("Failed to log completion message", error=str(e), traceback=traceback.format_exc())
                # Fallback logging without index_stats
                logger.info(
                    "Ingestion complete (basic)",
                    files_processed=successful_files,
                    total_files=len(text_files),
                    chunks_ingested=len(all_chunks)
                )
        else:
            logger.warning("No chunks were successfully processed")
    
    def ingest_s3_bucket(self, bucket_name: str, prefix: str = ""):
        
        try:
            
            s3_client = boto3.client(
                's3',
                region_name=self.settings.aws_region,
            )
            
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            
            if "Contents" not in response: 
                logger.warning("No objects found in S3 bucket", bucket=bucket_name, prefix=prefix)
                return
            
            objects = response["Contents"]
            
            text_objects = [obj for obj in objects if obj['Key'].endswith('.txt')]
            
            logger.info(
                "Found text files in S3",
                bucket=bucket_name,
                prefix=prefix,
                file_count=len(text_objects)
            )
            
            all_chunks = []
            
            successful_files = 0
            
            for obj in text_objects:
                try:
                    response = s3_client.get_object(Bucket=bucket_name, Key=obj['Key'])
                    
                    content = response['Body'].read().decode('utf-8')
                    
                    doc = Document(
                        page_content=content, 
                        metadata={
                            "source": f"s3://{bucket_name}/{obj['Key']}",
                        }
                    )
                    
                    chunks = self.text_splitter.split_documents([doc])
                    
                    processed_chunks = []
                    
                    for i, chunk in enumerate(chunks):
                        embedding = self.generate_embeddings(chunk.page_content)
                        doc_id = self.create_document_id(f"s3://{bucket_name}/{obj['Key']}", i)
                        
                        metadata = {
                            "text": chunk.page_content,
                            "source": f"s3://{bucket_name}/{obj['Key']}",
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "file_name": obj['Key'].split('/')[-1],
                            "file_size": obj['Size'],
                            "ingested_at": datetime.now(UTC).isoformat(),
                            "text_length": len(chunk.page_content)
                        }
                        
                        processed_chunks.append({
                            "id": doc_id,
                            "values": embedding,
                            "metadata": metadata
                        })
                    
                    all_chunks.extend(processed_chunks)
                    successful_files += 1
                    
                    logger.info(
                        "Processed S3 file",
                        file_key=obj['Key'],
                        chunks=len(processed_chunks)
                    )
            
                except Exception as e:
                    logger.error("Failed to process S3 file", file_key=obj['Key'], error=str(e))
                    continue
                
            if all_chunks:
                self.ingest_to_pinecone(all_chunks, 1)
                
                logger.info(
                    "S3 ingestion complete",
                    bucket=bucket_name,
                    prefix=prefix,
                    files_processed=successful_files,
                    total_files=len(text_objects),
                    chunks_ingested=len(all_chunks)
                )
            else:
                logger.warning("No chunks were successfully processed from S3")
            
        except Exception as e:
            logger.error("Failed to list S3 bucket objects", bucket=bucket_name, prefix=prefix)
            raise
            
    def _serialize_document(self):
        pass
    

def main():
    
    load_dotenv()
    
    doc_ingestor = DocumentIngestor()
    
    # proccessed_chunks = doc_ingestor.process_document(
    #     pathlib.Path("test_data.txt")
    # )
    
    # print(f"Processed \n{proccessed_chunks}\n chunks from the document.")
    
    # doc_ingestor.ingest_to_pinecone(proccessed_chunks, 2)
    
    doc_ingestor.ingest_s3_bucket(
        bucket_name="rag-aws-bucket-test",
        prefix="test/"
    )

if __name__ == "__main__":
    main()