from src.config import *


print("Testing configuration...")

print("AWS Region:", get_aws_region())

print("Bedrock Configuration:", get_bedrock_config())

print("Pinecone Configuration:", get_pipecone_config())

print("Settings:", get_settings().chunk_size)
print("Chunk Overlap:", get_settings().chunk_overlap)
