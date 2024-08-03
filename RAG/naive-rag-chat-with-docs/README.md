# Naive RAG Drawbacks

## Drawbacks of Naive RAG

### 1. Limited contextual understanding

Focus on keyword matching or basic semantic search (retrieving irrelevant or partially relevant documents).

For example, if the user asks a question about "Climate change with respect to the economy", the model may retrieve documents that contain the word "climate change" or "economy" but may not understand the relationship between the two concepts.

### 2. Inconsistent relevance and quality of retrieved documents

Varying in quality and relevance documents (poor-quality inputs for the gen model) because of the lack of ranking or filtering mechanism.

### 3. Poor integration between retrieval and generation

Retriever and generator to working in sync (unoptimized information flow) which may lead to irrelevant or repetitive responses.

### 4. Inefficient handling of Large-scale data

Scaling issues, take too long to find relevant docs, or miss critical info due to bad indexing.

### 5. Lack of Robustness and Adaptability

Not adaptable to changing contexts or user needs without significant manual intervention.

## Summary

We have

1. Retrieval challenges

Lead to the selection of misaligned or irrelevant chunks, therefore missing of crucial information.

2. Generation challenges

The model might struggle with hallucination and have issues with relevance, toxicity or bias in the generated text.
