# sentence transformers
from utils import prepare_db, get_vector_store, get_embed_model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


prepare_db()
vector_store = get_vector_store()


# Load Data
# TODO: Try out LlamaParse instead of PyMuPDFReader
from pathlib import Path
from llama_index.readers.file import PyMuPDFReader

loader = PyMuPDFReader()
# Documents here is a list of the pages of the file
# TODO: Iterate over all PDFs in `./data` and append `documents` list
documents = loader.load(file_path="./data/PhD_Arne-Reimers.pdf")


# Use Text Splitter to Split Documents
from llama_index.core.node_parser import SentenceSplitter

text_parser = SentenceSplitter(
    chunk_size=1024,
    # separator=" ",
)

text_chunks = []
# maintain relationship with source doc index, to help inject doc metadata in (3)
doc_idxs = []
for doc_idx, doc in enumerate(documents):
    cur_text_chunks = text_parser.split_text(doc.text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))


# Manually Construct Nodes from Text Chunks
from llama_index.core.schema import TextNode

nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc = documents[doc_idxs[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)


# Generate Embeddings for each Node
embed_model = get_embed_model()
for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding


# Load Nodes into a Vector Store
vector_store.add(nodes)
