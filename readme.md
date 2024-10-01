# Minimal RAG implmementation

This is based on ["Building RAG from Scratch (Open-source only!)"](https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/#building-rag-from-scratch-open-source-only) from the llamaindex docs, and inteded
to become a useful assistent for my thesis writing process by feeding the LLM
many recent dissertations from our group, giving it the ability to cross-reference
them for my own writing.


The application has two parts that are separated into two scripts.
First is the ingestion part where a PDF file is loaded,
chunked, transformed to a vector embedding and stored in a Postgres based vector store via LlamaIndex `PGVectorStore`.
This script is called `src/ingest_to_vector_db.py`.

The second part is the retrieval and inference, where the vector store is queried for the `top_k`
most similar nodes in the vector store and inference on the llm is run. The query on the vector store
is executed using LlamaIndex `VectorStoreQuery` and the llm inference is run via `LlamaCPP` referencing
the model url of a huggingface `gguf` model.


Ideas for experimentation:
* Implement abillity to work with official llama models from huggingface. Download fails atm.
* Change the PDF parser from PyMuPDFReader to LlamaParse
* Switch to a model finetuned to question-answer like behaviour, e.g. llama-instruct.
