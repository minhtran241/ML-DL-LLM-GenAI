import chromadb


def initialize_chroma_db(path, collection_name, embedding_function):
    chroma_client = chromadb.PersistentClient(path=path)
    collection = chroma_client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_function
    )
    return collection


def upsert_documents(collection, documents):
    for doc in documents:
        collection.upsert(
            ids=[doc["id"]],
            documents=[doc["text"]],
            embeddings=[doc["embedding"]],
            metadatas=[doc["metadata"]],
        )
