import pandas as pd
import os
import psycopg2
import inspect

from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext, Document
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.readers.file import DocxReader, PyMuPDFReader


class Indexer():

    def __init__(self,
                 db_name: str,
                 host: str,
                 password: str,
                 port: str,
                 user: str,
                 table_name: str,
                 embedding_model_name: str,
                 vector_size: int,
                 verbose: bool,
                 valid_extensions: list[str] = [".docx", ".pdf"],
                 ) -> None:
        self.db_name = db_name
        self.host = host
        self.password = password
        self.port = port
        self.user = user
        self.table_name = table_name
        self.embedding_model_name = embedding_model_name
        self.vector_size = vector_size
        self.valid_extensions = valid_extensions
        self.verbose = verbose

        self.EXTRACTOR_METHOD_SUFIX = "Reader"

    def clearDb(self):
        conn = psycopg2.connect(
            dbname="postgres",
            host=self.host,
            password=self.password,
            port=self.port,
            user=self.user,
        )
        conn.autocommit = True

        self.print(f"Deleting {self.table_name} table from {self.db_name} database")  # nopep8
        with conn.cursor() as c:
            c.execute(f"DROP DATABASE IF EXISTS {self.db_name}")
            c.execute(f"CREATE DATABASE {self.db_name}")

        conn = psycopg2.connect(
            dbname=self.db_name,
            host=self.host,
            password=self.password,
            port=self.port,
            user=self.user,
        )
        conn.autocommit = True

        with conn.cursor() as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
            c.execute(f"CREATE TABLE data_{self.table_name} (id bigserial PRIMARY KEY, text VARCHAR, metadata_ JSON, node_id VARCHAR, embedding VECTOR({self.vector_size}))")  # nopep8

    ######################
    #   Document Parsers
    ######################

    # This method could be simplified, but keeping methods for further specialization
    def docxReader(self, files: list, extension: str):
        parser = DocxReader()
        file_extractor = {extension: parser}
        documents = SimpleDirectoryReader(
            input_files=files, file_extractor=file_extractor
        ).load_data()
        return documents

    def pdfReader(self, files: list, extension: str):
        parser = PyMuPDFReader()
        file_extractor = {extension: parser}
        documents = SimpleDirectoryReader(
            input_files=files, file_extractor=file_extractor
        ).load_data()
        return documents

    ######################
    #   Helper methods
    ######################

    # This method is completely unnecessary, but keeping it for further compatibility
    def extractDocuments(self, path: str, recursive: bool = False):
        self.print(f"Reading documents with extensions {self.valid_extensions}")  # nopep8
        self.print(f"Path: {path}")  # nopep8
        if recursive:
            self.print(f"Searching recursively")  # nopep8

        documents = []
        file_list_dict = {ext: [] for ext in self.valid_extensions}
        for root, _, files in os.walk(path):
            for ext in self.valid_extensions:
                file_list_dict[ext].extend([os.path.join(
                    root, file) for file in files if file.endswith(ext)])
            if not recursive:
                break

        self.print("Obtained files from extensions")
        self.print(file_list_dict)

        for ext in self.valid_extensions:
            if len(file_list_dict[ext]):
                documentExtractor = getattr(
                    self, ext.replace(".", "")+self.EXTRACTOR_METHOD_SUFIX)
                documents.extend(documentExtractor(file_list_dict[ext], ext))

        return documents

    ######################
    #   Indexing
    ######################

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

    def indexDocumentsFromPath(self, path: str, recursive: bool = False):
        docs = self.extractDocuments(path, recursive)
        self.indexDocuments(docs)

    def print(self, text):
        if (self.verbose):
            print(text)
            print()


# Tokenizer?
# Settings.tokenzier = AutoTokenizer.from_pretrained(
#     "mistralai/Mixtral-8x7B-Instruct-v0.1"
# )

# set_global_tokenizer(
#     AutoTokenizer.from_pretrained(f"pcuenq/Llama-3.2-1B-Instruct-tokenizer").encode)  # must match your LLM
