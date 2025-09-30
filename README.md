# SePA Web-Search Pipeline

Implementation of the web-search pipeline from "SePA: A Search-enhanced Predictive Agent for Personalized Health Coaching" (IEEE BHI 2025).

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

