import os


def load_docs_from_dir(dir):
    docs = []
    for filename in os.listdir(dir):
        if filename.endswith(".txt"):
            with open(os.path.join(dir, filename), "r") as f:
                text = f.read()
                docs.append(
                    {
                        "id": filename,
                        "text": text,
                        "metadata": {"filename": filename, "source": "local"},
                    }
                )
    return docs


def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
    return chunks


def chunk_documents(docs, chunk_size=1000, overlap=200):
    chunked_docs = []
    for doc in docs:
        chunks = split_text(doc["text"], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            chunked_docs.append(
                {
                    "id": f"{doc['id']}_{i}",
                    "text": chunk,
                    "metadata": doc["metadata"],
                }
            )
    return chunked_docs
