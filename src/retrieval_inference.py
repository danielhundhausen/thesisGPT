from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.schema import NodeWithScore
from llama_index.core.query_engine import RetrieverQueryEngine
from typing import Optional

from utils import get_vector_store, get_embed_model
from retriever import VectorDBRetriever


# model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin"
model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"
model_url = "https://huggingface.co/cognitivecomputations/dolphin-2.9.4-llama3.1-8b-gguf/resolve/main/dolphin-2.9.4-llama3.1-8b-Q3_K_L.gguf"
model_url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct-Q4_K_L.gguf"


llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 0},
    verbose=True,
)


# TODO: Make this a cmd argument w/ argparse
query_str = "What is a possible extension of the Standard Model of particle physics?"
embed_model = get_embed_model()
query_embedding = embed_model.get_query_embedding(query_str)


# construct vector store query
query_mode = "default"  # "sparse" / "hybrid"
top_k = 2

vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=top_k, mode=query_mode
)

# returns a VectorStoreQueryResult
vector_store = get_vector_store()
query_result = vector_store.query(vector_store_query)
print(query_result.nodes[0].get_content())


# Parse result into set of nodes
nodes_with_scores = []
for index, node in enumerate(query_result.nodes):
    score: Optional[float] = None
    if query_result.similarities is not None:
        score = query_result.similarities[index]
    nodes_with_scores.append(NodeWithScore(node=node, score=score))


# Create Retriever object
retriever = VectorDBRetriever(
    vector_store, embed_model, query_mode=query_mode, similarity_top_k=top_k
)


# Plug into RetrieverQueryEngine --> Response
query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
response = query_engine.query(query_str)
print("********\n"*3)
print(query_str)
print("********\n")
print(str(response))
