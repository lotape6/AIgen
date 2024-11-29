from typing import Any, List, Optional
from llama_index.core import QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import BaseRetriever
from llama_index.llms.llama_cpp import LlamaCPP
from dotenv import load_dotenv
import os


# Shitty class distribution. Just momentary


class QueryEngineWrapper(BaseRetriever):
    """Retriever over a postgres vector store."""

    def __init__(self) -> None:
        """Init params."""
        super().__init__()

        load_dotenv()
        self.vector_store = PGVectorStore.from_params(
            database=os.getenv("DBNAME"),
            host=os.getenv("DBHOST"),
            password=os.getenv("DBPASSWORD"),
            port=os.getenv("DBPORT"),
            user=os.getenv("DBUSER"),
            table_name=os.getenv("TABLENAME"),
            # openai embedding dimension
            embed_dim=int(os.getenv("VECTOR_SIZE")),
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
                "hnsw_dist_method": "vector_cosine_ops",
            },
        )

        self.llm = LlamaCPP(
            model_path=os.getenv("MODEL_PATH"),
            temperature=1,
            max_new_tokens=3020,
            # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
            context_window=3900,
            # kwargs to pass to __call__()
            generate_kwargs={},
            # kwargs to pass to __init__()
            model_kwargs={"n_gpu_layers": 17},
            verbose=False,
        )

        self.query_engine = RetrieverQueryEngine.from_args(self, llm=self.llm)

        self.embed_model = HuggingFaceEmbedding(
            model_name=os.getenv("EMB_MODEL_NAME"))

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = self.embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=1,
            mode="default",
        )
        query_result = self.vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores

    def query(self, prompt):
        return self.query_engine.query(prompt)
