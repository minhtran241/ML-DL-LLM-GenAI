# Advanced RAG Techniques

Introduction to specific improvements to overcome the limitations of **Naive RAG**. Focus on enhancing the retrieval component of the RAG model.

## Pre-retrieval

-   Improvement of the **indexing structure** and **user's query**
-   Improves **data details**, **organizing indexes** better, adding **extra information**, **aligning things** correctly...

## Post-retrieval

Combine pre-retrieval data with the original query

**Re-ranking** the retrieved documents to highlight the most relevant ones

## Techniques

### Query Expansion (with generated answers)

_Check the code in the `expansion_answer.py` file_

-   Generate potential answers to the query (using an LLM) and to get **relevant** context.
-   Use cases:
    -   Information retrieval
    -   Question-answering systems
    -   E-commerce search engines
    -   Academic research

### Query Expansion (with multiple queries)

_Check the code in the `expansion_queries.py` file_

-   Use the LLM to generate **multiple additional queries** that might help getting the most relevant answer.
-   Use cases:
    -   Exploring data analysis
    -   Academic research
    -   Customer support
    -   Healthcare information systems
