from llama_index.core import SimpleDirectoryReader, StorageContext, Document
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from transformers import AutoTokenizer
from dotenv import load_dotenv
import textwrap
import psycopg2
import os
import pandas as pd
from pathlib import Path

load_dotenv()
clear_db = True if os.getenv("CLEARDB") == "True" else False
db_name = os.getenv("DBNAME")
host = os.getenv("DBHOST")
password = os.getenv("DBPASSWORD")
port = os.getenv("DBPORT")
user = os.getenv("DBUSER")
table_name = os.getenv("TABLENAME")

embedding_model_name = os.getenv("EMB_MODEL_NAME")
vector_size = os.getenv("VECTOR_SIZE")


def read_excel_as_documents(file_path):
    df = pd.read_excel(file_path)
    documents = []
    for _, row in df.iterrows():
        content = "\n".join(f"{col}: {row[col]}" for col in df.columns)
        documents.append(Document(text=content, metadata={"source": str(file_path)}))
    return documents

# Function to process all Excel files in a folder
def read_folder_as_documents(folder_path):
    documents = []
    folder = Path(folder_path)
    for file in folder.glob("*.xlsx"):  # Look for all Excel files
        documents.extend(read_excel_as_documents(file))
    return documents

documents = read_folder_as_documents("./resources/tuits")


print("Document ID:", documents[0].doc_id)

# Settings.tokenzier = AutoTokenizer.from_pretrained(
#     "mistralai/Mixtral-8x7B-Instruct-v0.1"
# )

# set_global_tokenizer(
#     AutoTokenizer.from_pretrained(f"pcuenq/Llama-3.2-1B-Instruct-tokenizer").encode)  # must match your LLM

embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
Settings.embed_model = embed_model

conn = psycopg2.connect(
    dbname="postgres",
    host=host,
    password=password,
    port=port,
    user=user,
)
conn.autocommit = True


if(clear_db):
    print("Clearing database")
    with conn.cursor() as c:
        c.execute(f"DROP DATABASE IF EXISTS {db_name}")
        c.execute(f"CREATE DATABASE {db_name}")
        c.execute(f"CREATE EXTENSION vector")
        c.execute(f"CREATE TABLE data_tuits (id bigserial PRIMARY KEY, text VARCHAR, metadata_ JSON, node_id VARCHAR, embedding VECTOR({int(vector_size)}))")
    


vector_store = PGVectorStore.from_params(
    database=db_name,
    host=host,
    password=password,
    port=port,
    user=user,
    table_name="tuits",
    embed_dim=int(vector_size),  # openai embedding dimension
    hnsw_kwargs={
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    },
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, show_progress=True
)
query_engine = index.as_query_engine()