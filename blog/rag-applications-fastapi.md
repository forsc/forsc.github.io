---
title: "Building Scalable RAG Applications with FastAPI and Vector Databases"
date: "2024-12-15"
category: "Machine Learning"
tags: ["RAG", "FastAPI", "Vector DB", "OpenAI"]
excerpt: "Exploring the architecture and implementation of production-ready Retrieval-Augmented Generation systems that can serve thousands of users efficiently. Learn about vector embeddings, semantic search, and feedback loops."
---

# Building Scalable RAG Applications with FastAPI and Vector Databases

## Introduction

Retrieval-Augmented Generation (RAG) has become a cornerstone technology for building AI applications that can provide accurate, contextual responses while maintaining factual grounding. In my recent work at AB-InBev, I've had the opportunity to build and deploy a RAG application that serves over 5,000 users internally, integrating SQL databases, internet search, vector databases, and sophisticated feedback loops.

## Architecture Overview

Our RAG system is built on several key components:

### 1. FastAPI Backend
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio

app = FastAPI(title="Enterprise RAG API", version="1.0.0")

class QueryRequest(BaseModel):
    question: str
    context_limit: int = 5
    include_web_search: bool = True

@app.post("/ask")
async def ask_question(request: QueryRequest):
    # RAG pipeline implementation
    pass
```

### 2. Vector Database Integration
We use a combination of vector databases to handle different types of content:
- **Document Embeddings**: Company policies, procedures, and knowledge base
- **SQL Query Embeddings**: Pre-built query patterns for database operations
- **Web Content**: Cached and indexed external resources

### 3. Multi-Source Retrieval
The system retrieves relevant information from:
- Internal vector database (company documents)
- SQL database queries (real-time data)
- Web search results (current information)
- User feedback history (personalization)

## Implementation Details

### Vector Embedding Strategy
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed_documents(self, documents):
        embeddings = self.model.encode(documents)
        return embeddings
    
    def similarity_search(self, query, embeddings, top_k=5):
        query_embedding = self.model.encode([query])
        similarities = np.dot(embeddings, query_embedding.T)
        top_indices = np.argsort(similarities.flatten())[-top_k:]
        return top_indices[::-1]
```

### FastAPI + OpenAI Agent Integration
```python
from openai import OpenAI
import agno  # Custom agent framework

class RAGAgent:
    def __init__(self):
        self.client = OpenAI()
        self.retriever = DocumentRetriever()
    
    async def process_query(self, question: str):
        # 1. Retrieve relevant documents
        relevant_docs = await self.retriever.search(question)
        
        # 2. Format context
        context = self.format_context(relevant_docs)
        
        # 3. Generate response
        response = await self.generate_response(question, context)
        
        # 4. Log for feedback loop
        await self.log_interaction(question, response)
        
        return response
```

## Kubernetes Deployment

Our system runs on Azure Kubernetes Service (AKS) with the following configuration:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: ragapp:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
```

## Performance Optimizations

### 1. Async Processing
All I/O operations are handled asynchronously to maximize throughput:
```python
async def parallel_retrieval(query):
    tasks = [
        retrieve_from_vector_db(query),
        search_sql_database(query),
        fetch_web_results(query)
    ]
    results = await asyncio.gather(*tasks)
    return merge_results(results)
```

### 2. Caching Strategy
- **Vector embeddings**: Cached for 24 hours
- **SQL results**: Cached for 1 hour with invalidation
- **Web search**: Cached for 6 hours

### 3. Rate Limiting and Load Balancing
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/ask")
@limiter.limit("10/minute")
async def ask_question(request: Request, query: QueryRequest):
    # Rate-limited endpoint
    pass
```

## Feedback Loop Implementation

One of the most critical aspects of our RAG system is the feedback mechanism:

```python
class FeedbackProcessor:
    def __init__(self):
        self.feedback_db = FeedbackDatabase()
    
    async def process_feedback(self, query_id: str, rating: int, comments: str):
        # Store feedback
        await self.feedback_db.store(query_id, rating, comments)
        
        # Update retrieval weights
        if rating < 3:
            await self.adjust_retrieval_weights(query_id, decrease=True)
        elif rating > 4:
            await self.adjust_retrieval_weights(query_id, increase=True)
```

## Results and Impact

Since deployment, our RAG application has achieved:
- **5,000+ active users** across the organization
- **95% user satisfaction** rate based on feedback
- **40% reduction** in support ticket volume
- **3-second average** response time
- **99.9% uptime** on Kubernetes infrastructure

## Key Learnings

### 1. Context Window Management
Managing the context window effectively is crucial. We implement a smart truncation strategy that prioritizes:
- Most relevant documents (by similarity score)
- Most recent information
- User-specific context

### 2. Multi-Modal Integration
Combining structured (SQL) and unstructured (documents) data sources requires careful orchestration and consistent embedding strategies.

### 3. User Feedback is Gold
The feedback loop has been instrumental in improving accuracy. Users provide context that pure similarity search cannot capture.

## Future Enhancements

We're planning several improvements:
- **Fine-tuned embeddings** on domain-specific data
- **Graph RAG** for better relationship understanding  
- **Multi-language support** for global deployment
- **Advanced reasoning** with chain-of-thought prompting

## Conclusion

Building a production RAG system requires thoughtful architecture, robust infrastructure, and continuous iteration based on user feedback. The combination of FastAPI's async capabilities, Kubernetes scalability, and OpenAI's latest models provides a solid foundation for enterprise-grade AI applications.

The key to success is starting simple, measuring everything, and iterating based on real user needs. Our system serves thousands of users daily and continues to evolve based on their feedback and changing requirements.

---

*Have questions about implementing RAG systems? Feel free to reach out on [LinkedIn](https://linkedin.com/in/rahulbhow) or check out my other posts on machine learning engineering.* 