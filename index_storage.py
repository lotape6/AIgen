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


class Indexer():

    def __init__(self,
                 db_name: str = os.getenv("DBNAME"),
                 host: str = os.getenv("DBHOST"),
                 password: str = os.getenv("DBPASSWORD"),
                 port: str = os.getenv("DBPORT"),
                 user: str = os.getenv("DBUSER"),
                 table_name: str = os.getenv("TABLENAME"),
                 embedding_model_name: str = os.getenv("EMB_MODEL_NAME"),
                 vector_size: int = os.getenv("VECTOR_SIZE")
                 ) -> None:
        self.db_name = db_name
        self.host = host
        self.password = password
        self.port = port
        self.user = user
        self.table_name = table_name
        self.embedding_model_name = embedding_model_name
        self.vector_size = vector_size

        self.valid_extensions = [".xslx"]

    def clearDb(self):
        conn = psycopg2.connect(
            dbname="postgres",
            host=self.host,
            password=self.password,
            port=self.port,
            user=self.user,
        )
        conn.autocommit = True

        print("Clearing database")
        with conn.cursor() as c:
            c.execute(f"DROP DATABASE IF EXISTS {self.db_name}")
            c.execute(f"CREATE DATABASE {self.db_name}")
            c.execute(f"CREATE EXTENSION vector")
            c.execute(f"CREATE TABLE data_tuits (id bigserial PRIMARY KEY, text VARCHAR, metadata_ JSON, node_id VARCHAR, embedding VECTOR({
                      self.vector_size})")

    def readExcelAsDocument(self, file_path):
        df = pd.read_excel(file_path)
        documents = []
        for _, row in df.iterrows():
            content = "\n".join(f"{col}: {row[col]}" for col in df.columns)
            documents.append(Document(text=content, metadata={
                "source": str(file_path)}))
        return documents

    def xslxExtractDocuments(self, files: list):
        documents = []
        for file in files:
            documents.extend(self.readExcelAsDocument(file))
        return documents

    def pdfExtractDocuments(slef, files: list):
        documents = []

        return documents

    def extractDocuments(self, path: str, recursive: bool = False):
        documents = []
        file_list_dict = {ext: [] for ext in self.valid_extensions}
        for root, _, files in os.walk(path):
            for ext in self.valid_extensions:
                file_list_dict[ext].append([os.path.join(
                    root, file) for file in files if file.endswith(ext)])

            if not recursive:
                break

        for ext in self.valid_extensions:
            documentExtractor = getattr(
                self, ext.replace(".", "")+"ExtractDocuments")
            documents.extend(documentExtractor(file_list_dict[ext]))

        return documents

    def indexDocuments(self, documents: list):
        embed_model = HuggingFaceEmbedding(
            model_name=self.embedding_model_name)
        Settings.embed_model = embed_model

        vector_store = PGVectorStore.from_params(
            database=self.db_name,
            host=self.host,
            password=self.password,
            port=self.port,
            user=self.user,
            table_name="tuits",
            embed_dim=self.vector_size,
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
                "hnsw_dist_method": "vector_cosine_ops",
            },
        )

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, show_progress=True
        )


# Tokenizer?
# Settings.tokenzier = AutoTokenizer.from_pretrained(
#     "mistralai/Mixtral-8x7B-Instruct-v0.1"
# )

# set_global_tokenizer(
#     AutoTokenizer.from_pretrained(f"pcuenq/Llama-3.2-1B-Instruct-tokenizer").encode)  # must match your LLM
