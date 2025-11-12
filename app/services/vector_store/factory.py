from typing import Optional
from pymongo import MongoClient
from langchain_core.embeddings import Embeddings

from .async_pg_vector import AsyncPgVector
from .atlas_mongo_vector import AtlasMongoVector
from .extended_pg_vector import ExtendedPgVector
from .qdrant_vector import QdrantVector


def get_vector_store(
    connection_string: str,
    embeddings: Embeddings,
    collection_name: str,
    mode: str = "sync",
    search_index: Optional[str] = None
):
    if mode == "sync":
        return ExtendedPgVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    elif mode == "async":
        return AsyncPgVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    elif mode == "atlas-mongo":
        mongo_db = MongoClient(connection_string).get_database()
        mong_collection = mongo_db[collection_name]
        return AtlasMongoVector(
            collection=mong_collection, embedding=embeddings, index_name=search_index
        )
    elif mode == "qdrant":
        # Import QDRANT_API_KEY from config
        from app.config import QDRANT_API_KEY
        return QdrantVector(
            url=connection_string,
            api_key=QDRANT_API_KEY,
            collection_name=collection_name,
            embeddings=embeddings,
        )
    else:
        raise ValueError("Invalid mode specified. Choose 'sync', 'async', 'atlas-mongo', or 'qdrant'.")