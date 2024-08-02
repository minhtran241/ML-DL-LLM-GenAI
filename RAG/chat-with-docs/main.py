from config import load_config
from embedding import configure_genai, genai_embed
from database import initialize_chroma_db, upsert_documents
from docproc import load_docs_from_dir, chunk_documents
from qa import query_docs, generate_response


def main():
    google_api_key = load_config()
    model, google_ef = configure_genai(google_api_key)

    collection = initialize_chroma_db(
        "chroma_db.sqlite", "document_qa_collection", google_ef
    )

    docs = load_docs_from_dir("./news_articles")
    print(f"Loaded {len(docs)} documents")

    chunked_docs = chunk_documents(docs)
    print(f"Split documents into {len(chunked_docs)} chunks")

    for doc in chunked_docs:
        doc["embedding"] = genai_embed(doc["text"], model)

    upsert_documents(collection, chunked_docs)
    print(f"Added {len(chunked_docs)} document chunks to the collection in Chroma DB")

    question = "tell me about AI replacing TV writers strike."
    relevant_chunks = query_docs(collection, question)
    answer = generate_response(model, question, relevant_chunks)

    print(answer)


if __name__ == "__main__":
    main()
