# Connect to PostgresDB
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import psycopg2


db_name = "vector_db"
host = "localhost"
password = "redondita"
port = "5432"
user = "admin"


def prepare_db() -> None:
    # conn = psycopg2.connect(connection_string)
    conn = psycopg2.connect(
        dbname="postgres",
        host=host,
        password=password,
        port=port,
        user=user,
    )
    conn.autocommit = True

    with conn.cursor() as c:
        c.execute(f"DROP DATABASE IF EXISTS {db_name}")
        c.execute(f"CREATE DATABASE {db_name}")


def get_vector_store():
    vector_store = PGVectorStore.from_params(
        database=db_name,
        host=host,
        password=password,
        port=port,
        user=user,
        table_name="llama2_paper",
        embed_dim=384,  # openai embedding dimension
    )
    return vector_store 


def get_embed_model():
    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
