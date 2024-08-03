import google.generativeai as genai
from chromadb.utils import embedding_functions


def configure_genai(api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=api_key
    )
    return model, embedding_function


def genai_embed(text, model):
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document",
    )
    embedding = result["embedding"]
    return embedding
