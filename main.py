from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from time import time
from llama_index.llms.openai import OpenAI
from llama_index.llms.llama_cpp import LlamaCPP
from dotenv import load_dotenv
import os

load_dotenv()
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")



llm = LlamaCPP(
    model_path=os.getenv("MODEL_PATH"),
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    model_kwargs={"n_gpu_layers": 29},
    verbose=True,
)

while True:
    prompt = input("Enter a prompt ")
    response = llm.complete(prompt)
    print(response.text)