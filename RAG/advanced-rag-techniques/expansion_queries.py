from helper_utils import (
    project_embeddings,
    # word_wrap,
)  # Import utility functions for embedding projection and text wrapping
from dotenv import (
    load_dotenv,
)  # Import function to load environment variables from a .env file
from pypdf import PdfReader  # Import PdfReader for reading PDF files
import os  # Import os module for interacting with the operating system
import chromadb  # Import ChromaDB for document embedding and retrieval
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)  # Import embedding function from ChromaDB
import google.generativeai as genai  # Import Google Generative AI module
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)  # Import text splitters for chunking text
import umap  # Import UMAP for dimensionality reduction
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Load environment variables from the .env file
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


# Define a function to generate multiple related queries using Google Generative AI
def generate_multi_query(query, model="gemini-1.5-flash"):
    prompt = """
    You are a knowledgeable financial research assistant. 
    Your users are inquiring about an annual report. 
    For the given question, propose up to five related questions to assist them in finding the information they need. 
    Provide concise, single-topic questions (without compounding sentences) that cover various aspects of the topic. 
    Ensure each question is complete and directly related to the original inquiry. 
    List each question on a separate line without numbering.
    """
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
    # Convert to list of strings
    return response.text.split("\n")


# Generate multiple related queries for the original query
original_query = (
    "What details can you provide about the factors that led to revenue growth?"
)
aug_queries = generate_multi_query(original_query)

# Concatenate the original query with the augmented queries
joint_query = [
    original_query
] + aug_queries  # original query is in a list because Chroma can handle multiple queries

print("\n\n".join(joint_query))

# Query the Chroma collection with the original and augmented queries
results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)
retrieved_docs = results["documents"]

print(f"\nTotal number of retrieved documents: {len(retrieved_docs)}")

# Deduplicate the retrieved documents
i = 0
unique_docs = set()
for docs in retrieved_docs:
    for doc in docs:
        i += 1
        unique_docs.add(doc)

print(f"\nTotal number of retrieved documents before deduplication: {i}")
print(f"\nNumber of unique retrieved documents: {len(unique_docs)}")

# Output the unique retrieved documents (commented out to avoid cluttering the output)
# for i, docs in enumerate(retrieved_docs):
#     print(f"Query: {joint_query[i]}")
#     print("")
#     print("Retrieved Documents:")
#     for doc in docs:
#         print(word_wrap(doc))
#         print("")
#     print("-" * 100)

# Get the embeddings of the retrieved documents
embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

# Visualize the embeddings using UMAP
original_query_embedding = ef([original_query])
augmented_query_embeddings = ef(joint_query)

project_original_query = project_embeddings(original_query_embedding, umap_transform)
project_augmented_queries = project_embeddings(
    augmented_query_embeddings, umap_transform
)

retrieved_embeddings = results["embeddings"]
result_embeddings = [item for sublist in retrieved_embeddings for item in sublist]

projected_result_embeddings = project_embeddings(result_embeddings, umap_transform)

# Plot the projected query and retrieved documents in the embedding space
plt.figure()
plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)
plt.scatter(
    project_augmented_queries[:, 0],
    project_augmented_queries[:, 1],
    s=150,
    marker="X",
    color="orange",
)
plt.scatter(
    projected_result_embeddings[:, 0],
    projected_result_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    project_original_query[:, 0],
    project_original_query[:, 1],
    s=150,
    marker="X",
    color="r",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()  # Display the plot
