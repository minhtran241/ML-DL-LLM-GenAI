from helper_utils import project_embeddings
from dotenv import load_dotenv
from pypdf import PdfReader
import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import google.generativeai as genai
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import umap
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

# Configure the Google Generative AI with the API key from environment variables
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize the PDF reader and extract text from each page, filtering out empty strings
reader = PdfReader(stream="./data/microsoft-annual-report.pdf")
pdf_texts = [
    page.extract_text().strip() for page in reader.pages if page.extract_text()
]

# Step 1: Split the text into smaller chunks using character-based splitting
character_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        "\r\n",
        "\r",
        "\t",
        ". ",
        " ",
        "",
    ],  # Define separators for splitting text
    chunk_size=1000,  # Define the size of each chunk
    chunk_overlap=0,  # Define the overlap between chunks
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

# Print the summary statistics of the character-based chunks
print(
    f"\nTotal number of character-based chunks: {len(character_split_texts)} | Total number of pages: {len(pdf_texts)} | Total number of characters: {sum(len(text) for text in pdf_texts)}"
)

# Step 2: Split the text into smaller chunks using token-based splitting
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0,  # Define the overlap between chunks
    tokens_per_chunk=256,  # Define the number of tokens per chunk
)
token_split_texts = [
    chunk for text in character_split_texts for chunk in token_splitter.split_text(text)
]

# Print the summary statistics of the token-based chunks
print(
    f"\nTotal number of token-based chunks: {len(token_split_texts)} | Total number of pages: {len(pdf_texts)} | Total number of characters: {sum(len(text) for text in pdf_texts)}"
)

# Initialize the embedding function using SentenceTransformer
ef = SentenceTransformerEmbeddingFunction()

# Initialize the Chroma client and create a collection for storing the document embeddings
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "microsoft_annual_report", embedding_function=ef
)

# Extract embeddings for each chunk and add them to the Chroma collection
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
count = chroma_collection.count()  # Get the count of documents in the collection

# Query the Chroma collection with a specific question
query = "What was the total revenue for the year?"
results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_docs = results["documents"][0]


# Function to augment the retrieved documents using Google Generative AI
def augment_query_generated(query, model="gemini-1.5-flash"):
    """
    Generate an augmented query using Google Generative AI.

    Parameters:
    query (str): The original query to be augmented.
    model (str): The model to use for generating the augmented query.

    Returns:
    str: The augmented query.
    """
    prompt = """You are a helpful expert financial research assistant. 
    Provide an example answer to the given question, that might be found in a document like an annual report."""

    # Initialize the generative model and start a chat
    client = genai.GenerativeModel(model)
    chat = client.start_chat(
        history=[
            {"role": "user", "parts": [prompt]},
            {
                "role": "model",
                "parts": ["Sure, I can help with that. Please provide the question."],
            },
        ]
    )

    # Send the query to the chat and return the response
    response = chat.send_message(query)
    return response.text


# Generate an augmented query based on the original query
original_query = "What was the total profit for the year, and how does it compare to the previous year?"
hypothetical_answer = augment_query_generated(original_query)
joint_query = f"{original_query} {hypothetical_answer}"

# Query the Chroma collection with the augmented query
results = chroma_collection.query(
    query_texts=[joint_query], n_results=5, include=["documents", "embeddings"]
)

# Get embeddings from the Chroma collection
embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]

# Fit the UMAP model on the dataset embeddings
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

# Get the embeddings for the original and augmented queries
retrieved_embeddings = results["embeddings"][0]
original_query_embedding = ef([original_query])
augmented_query_embedding = ef([joint_query])

# Project the embeddings into a 2D space using UMAP
projected_original_query_embedding = project_embeddings(
    original_query_embedding, umap_transform
)
projected_augmented_query_embedding = project_embeddings(
    augmented_query_embedding, umap_transform
)
projected_retrieved_embeddings = project_embeddings(
    retrieved_embeddings, umap_transform
)

# Visualize the embeddings using UMAP
plt.figure()

# Plot the projected query and retrieved documents in the embedding space
plt.scatter(
    projected_dataset_embeddings[:, 0],  # [:, 0] means all rows, column 0
    projected_dataset_embeddings[:, 1],  # [:, 1] means all rows, column 1
    s=10,
    color="gray",
)
plt.scatter(
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="X",
    color="r",
)
plt.scatter(
    projected_augmented_query_embedding[:, 0],
    projected_augmented_query_embedding[:, 1],
    s=150,
    marker="X",
    color="orange",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()  # Display the plot
