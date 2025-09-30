# SePA Web Search Agent

Web-search RAG system for athletic health information retrieval with real-time search capabilities. Exact implementation tested and explained in the paper. 

## Overview

This system combines real-time web search across trusted health domains with intelligent caching and LLM-based response generation to provide evidence-based health recommendations.

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
