# SePA Web Search Agent

Web-search RAG system for athletic health information retrieval with real-time search capabilities.

## About SePA

This repository contains the **web search and retrieval-augmented generation component** from SePA (Search-enhanced Predictive AI Agent), described in our IEEE paper:

> **Abstract:** This paper introduces SePA (Search-enhanced Predictive AI Agent), a novel LLM health coaching system that integrates personalized machine learning and retrieval-augmented generation to deliver adaptive, evidence-based guidance. SePA combines: (1) Individualized models predicting daily stress, soreness, and injury risk from wearable sensor data (28 users, 1260 data points); and (2) A retrieval module that grounds LLM-generated feedback in expert-vetted web content to ensure contextual relevance and reliability. Our predictive models, evaluated with rolling-origin cross-validation and group 4-fold cross-validation show that personalized models outperform generalized baselines. In a pilot expert study (n=4), SePA's retrieval-based advice was preferred over a non-retrieval baseline, yielding meaningful practical effect (Cliff's δ = 0.3, p = 0.05). We also quantify latency performance trade-offs between response quality and speed, offering a transparent blueprint for next-generation, trustworthy personal health informatics systems.

**This repository implements component (2)** - the retrieval and web search system that grounds LLM responses in authoritative health content.

## Overview

This system combines real-time web search across trusted health domains with intelligent caching and LLM-based response generation to provide evidence-based health recommendations.

## Architecture

- **web_rag.py** - Main RAG orchestrator
- **search.py** - Web search and content processing
- **cache.py** - Caching system
- **models.py** - Model initializations
- **config.yaml** - Configuration and trusted domains

## Usage

```python
from web_rag import create_rag
import asyncio

async def main():
    rag = create_rag()

    query = "How to improve VO2 max for basketball players"
    user_context = {"age": 20, "sport": "basketball"}

    vector_store = await rag.process_query(query, user_context)
    response = await rag.rag_generate_response(
        query, user_context, [], vector_store=vector_store
    )

    print(response['answer'])

asyncio.run(main())
```

## Setup

```bash
pip install -r requirements.txt

export OPENAI_API_KEY="your_key"
export GOOGLE_API_KEY="your_key"
export GOOGLE_CSE_ID="your_cse_id"
```
