// Blog data embedded as JavaScript to avoid CORS issues
window.blogData = [
    {
        id: "rag-applications-fastapi",
        title: "Building Scalable RAG Applications with FastAPI and Vector Databases",
        date: "2024-12-15",
        category: "Machine Learning",
        tags: ["RAG", "FastAPI", "Vector DB", "OpenAI"],
        excerpt: "Exploring the architecture and implementation of production-ready Retrieval-Augmented Generation systems that can serve thousands of users efficiently. Learn about vector embeddings, semantic search, and feedback loops.",
        content: `
# Building Scalable RAG Applications with FastAPI and Vector Databases

## Introduction

Retrieval-Augmented Generation (RAG) has become a cornerstone technology for building AI applications that can provide accurate, contextual responses while maintaining factual grounding. In my recent work at AB-InBev, I've had the opportunity to build and deploy a RAG application that serves over 5,000 users internally, integrating SQL databases, internet search, vector databases, and sophisticated feedback loops.

## Architecture Overview

Our RAG system is built on several key components:

### 1. FastAPI Backend
\`\`\`python
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
\`\`\`

### 2. Vector Database Integration
We use a combination of vector databases to handle different types of content:
- **Document Embeddings**: Company policies, procedures, and knowledge base
- **SQL Query Embeddings**: Pre-built query patterns for database operations
- **Web Content**: Cached and indexed external resources

## Performance Results

Since deployment, our RAG application has achieved:
- **5,000+ active users** across the organization
- **95% user satisfaction** rate based on feedback
- **40% reduction** in support ticket volume
- **3-second average** response time
- **99.9% uptime** on Kubernetes infrastructure

## Conclusion

Building a production RAG system requires thoughtful architecture, robust infrastructure, and continuous iteration based on user feedback. The combination of FastAPI's async capabilities, Kubernetes scalability, and OpenAI's latest models provides a solid foundation for enterprise-grade AI applications.

---

*Have questions about implementing RAG systems? Feel free to reach out on [LinkedIn](https://linkedin.com/in/rahulbhow) or check out my other posts on machine learning engineering.*
        `
    },
    {
        id: "recommendation-systems-scale",
        title: "Recommendation Systems at Scale: From XGBoost to Neural Collaborative Filtering",
        date: "2024-11-28",
        category: "Data Science", 
        tags: ["Recommendations", "XGBoost", "Neural Networks", "E-commerce"],
        excerpt: "A deep dive into building recommendation algorithms for e-commerce platforms. Compare traditional ML approaches with modern deep learning techniques and their real-world performance metrics.",
        content: `
# Recommendation Systems at Scale: From XGBoost to Neural Collaborative Filtering

## The Challenge of Scale

At AB-InBev, we process millions of transactions daily across hundreds of products and thousands of customers. Building recommendation systems that can provide relevant suggestions in real-time while driving business outcomes requires a sophisticated approach that balances accuracy, latency, and explainability.

## Our Multi-Algorithm Approach

### 1. XGBoost for Feature-Rich Predictions

We start with XGBoost as our baseline model because of its excellent performance on tabular data and interpretability:

\`\`\`python
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd

class XGBoostRecommender:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    
    def predict_score(self, user_id, product_id):
        features = self.prepare_features(user_id, product_id)
        return self.model.predict(features)[0]
\`\`\`

## Performance Metrics & A/B Testing

### Online A/B Testing Results

| Algorithm | CTR Improvement | Conversion Rate | Revenue Uplift |
|-----------|----------------|-----------------|----------------|
| XGBoost Only | +12% | +8% | +15% |
| LightFM Only | +18% | +11% | +22% |
| NCF Only | +16% | +9% | +19% |
| **Ensemble** | **+25%** | **+16%** | **+31%** |

## Conclusion

Building production recommendation systems requires balancing multiple objectives: accuracy, latency, explainability, and business impact. Our ensemble approach combining XGBoost, LightFM, and neural collaborative filtering has delivered significant business value while maintaining the flexibility to adapt to changing user preferences.

---

*Interested in recommendation systems? Connect with me on [LinkedIn](https://linkedin.com/in/rahulbhow) for more insights on ML engineering at scale.*
        `
    },
    {
        id: "distributed-training-ray",
        title: "Distributed Model Training with Ray on Databricks: 80% Efficiency Gains",
        date: "2024-11-10",
        category: "MLOps",
        tags: ["Ray", "Databricks", "Distributed Computing", "MLOps"],
        excerpt: "How we achieved massive performance improvements in model training using Ray's distributed computing framework. Includes code examples and performance benchmarks from production deployments.",
        content: `
# Distributed Model Training with Ray on Databricks: 80% Efficiency Gains

## The Scale Challenge

Training machine learning models on massive datasets requires more than just better algorithms—it demands efficient distributed computing. At AB-InBev, we process terabytes of transaction data to train recommendation models, and our traditional single-node training was becoming a bottleneck.

Enter Ray: a unified framework for scaling Python and AI applications. By implementing Ray on Databricks, we achieved an 80% improvement in training efficiency while maintaining model quality.

## Why Ray + Databricks?

### The Perfect Combination
- **Ray**: Handles distributed computing complexities
- **Databricks**: Provides managed Spark infrastructure and MLflow integration
- **Combined**: Seamless scaling from laptop to cluster

\`\`\`python
import ray
from ray import train
from ray.train import Trainer
from ray.train.integrations.mlflow import MLflowLoggerCallback

# Initialize Ray on Databricks cluster
ray.init(address="ray://head-node:10001")
print(f"Ray cluster: {ray.cluster_resources()}")
\`\`\`

## Performance Results

### Benchmark Comparison

| Metric | Single Node | Ray Distributed | Improvement |
|--------|-------------|-----------------|-------------|
| Training Time | 45 minutes | 9 minutes | **80% faster** |
| Memory Usage | 32 GB | 8 GB per node | **75% reduction** |
| Hyperparameter Trials | 10/hour | 120/hour | **12x throughput** |
| Model Quality (RMSE) | 0.245 | 0.242 | **1.2% better** |
| Cost per Training Run | $15 | $8 | **47% savings** |

## Conclusion

Ray on Databricks has transformed our ML training pipeline, delivering:
- **80% faster training** through efficient parallelization
- **Cost savings** through better resource utilization  
- **Better models** via extensive hyperparameter search
- **Simplified operations** with managed infrastructure

The combination of Ray's distributed computing capabilities and Databricks' managed platform provides a powerful foundation for scaling ML workloads. Start small, measure everything, and scale incrementally based on your specific bottlenecks.

---

*Want to learn more about distributed ML? Connect with me on [LinkedIn](https://linkedin.com/in/rahulbhow) for insights on scaling ML systems.*
        `
    }
]; 