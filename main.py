from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from time import time
from llama_index.llms.openai import OpenAI
from llama_index.llms.llama_cpp import LlamaCPP
from dotenv import load_dotenv
import os

load_dotenv()
embed_model = HuggingFaceEmbedding(model_name=os.getenv("EMB_MODEL_NAME"))

llm = LlamaCPP(
    model_path=os.getenv("MODEL_PATH"),
    temperature=0.0001,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    model_kwargs={"n_gpu_layers": 17},
    verbose=False,
)

db_name = os.getenv("DBNAME")
host = os.getenv("DBHOST")
password = os.getenv("DBPASSWORD")
port = os.getenv("DBPORT")
user = os.getenv("DBUSER")
table_name = os.getenv("TABLENAME")

from llama_index.vector_stores.postgres import PGVectorStore

vector_store = PGVectorStore.from_params(
    database=db_name,
    host=host,
    password=password,
    port=port,
    user=user,
    table_name="tuits",
    embed_dim=int(os.getenv("VECTOR_SIZE")),  # openai embedding dimension
    hnsw_kwargs={
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    },
)


# Build retrieval pipeline (check https://docs.llamaindex.ai/en/stable/examples/low_level/retrieval/)
from llama_index.core.vector_stores import VectorStoreQuery

query_mode = "default"
# query_mode = "sparse"
# query_mode = "hybrid"


from llama_index.core.schema import NodeWithScore
from typing import Optional
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from typing import Any, List

class VectorDBRetriever(BaseRetriever):
    """Retriever over a postgres vector store."""

    def __init__(
        self,
        vector_store: PGVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 4,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores

retriever = VectorDBRetriever(
    vector_store, embed_model, query_mode=query_mode, similarity_top_k=1
)

from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

while True:
    query_str = input("Enter a prompt: ")
    
    response = query_engine.query(query_str)
    print(str(response))
    print("----------------------------------------------------------------------")