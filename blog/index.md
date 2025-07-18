# Blog Posts Index

This directory contains all blog posts for the portfolio website. Each post is written in Markdown with YAML frontmatter.

## Available Posts

### 1. RAG Applications with FastAPI
- **File**: `rag-applications-fastapi.md`
- **Date**: 2024-12-15
- **Category**: Machine Learning
- **Topics**: RAG, FastAPI, Vector Databases, OpenAI

### 2. Recommendation Systems at Scale
- **File**: `recommendation-systems-scale.md`
- **Date**: 2024-11-28
- **Category**: Data Science
- **Topics**: XGBoost, Neural Networks, E-commerce

### 3. Distributed Training with Ray
- **File**: `distributed-training-ray.md`
- **Date**: 2024-11-10
- **Category**: MLOps
- **Topics**: Ray, Databricks, Distributed Computing

## Adding New Posts

1. Create a new `.md` file in this directory
2. Use the following frontmatter structure:

```yaml
---
title: "Your Post Title"
date: "YYYY-MM-DD"
category: "Category Name"
tags: ["tag1", "tag2", "tag3"]
excerpt: "Brief description for the blog card"
---
```

3. Add the filename to the `blogList` array in `blog-reader.js`
4. The blog reader will automatically load and display your post

## Frontmatter Fields

- **title**: The title of your blog post (required)
- **date**: Publication date in YYYY-MM-DD format (required)
- **category**: Main category for color coding (required)
- **tags**: Array of tags for filtering and display (required)
- **excerpt**: Brief description shown on blog cards (required)

## Supported Markdown Features

- Headers (H1-H6)
- Code blocks with syntax highlighting
- Inline code
- Bold and italic text
- Links
- Lists (unordered)
- Tables
- Line breaks and paragraphs

## Categories

Current categories with their color schemes:
- Machine Learning (Blue gradient)
- Data Science (Pink gradient)
- MLOps (Cyan gradient)
- AI/ML (Green gradient)
- Career (Orange gradient)
- Technical (Purple gradient)

You can add new categories - they will automatically get assigned colors from the gradient palette defined in `script.js`. 