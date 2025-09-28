# SePA Web-Search Pipeline

Implementation of the web-search pipeline from "SePA: A Search-enhanced Predictive Agent for Personalized Health Coaching" (IEEE BHI 2025).

## Abstract
This paper introduces SePA (Search-enhanced Predictive AI Agent), a novel LLM health coaching system that integrates personalized machine learning and retrieval-augmented generation to deliver adaptive, evidence-based guidance. SePA combines: (1) Individualized models predicting daily stress, soreness, and injury risk from wearable sensor data (28 users, 1260 data points); and (2) A retrieval module that grounds LLM-generated feedback in expert-vetted web content to ensure contextual relevance and reliability. Our predictive models, evaluated with rolling-origin cross-validation and group 4-fold cross-validation show that personalized models outperform generalized baselines. In a pilot expert study (n=4), SePA's retrieval-based advice was preferred over a non-retrieval baseline, yielding meaningful practical effect (Cliff’s $\delta = 0.3$, *p* = 0.05). We also quantify latency performance trade-offs between response quality and speed, offering a transparent blueprint for next-generation, trustworthy personal health informatics systems.

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
