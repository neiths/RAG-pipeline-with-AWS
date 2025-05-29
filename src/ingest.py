import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from dotenv import load_dotenv

import os
import json

import logging

from config import get_settings

logger = logging.getLogger(__name__)

class DocumentIngestor:
    
    def __init__(self):
        """Initialize ingestor with AWS and pipecone clients"""
        self.settings = get_settings()
        self._setup_clients()
           
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
        pass
    
    def generate_embeddings(self, text: str):
        pass
    
    def create_document_id(self, text):
        pass
    
    def process_document(self, text: str):
        pass

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
    
    # input_text = "Hello from AWs embedding!"
    
    # payload = {
    #     'inputText': input_text,
    # }
    
    # model_id = "amazon.titan-embed-text-v2:0"
    
    # response = doc_ingestor.bedrock_client.invoke_model(
    #     modelId=model_id,
    #     body=json.dumps(payload),
    # )
    
    # print("Response: ", response, "\n")
    

    # response_body = json.loads(response['body'].read())
    # embedding = response_body.get('embedding')
    
    # print("Embedding: ", embedding, "\n")

if __name__ == "__main__":
    main()