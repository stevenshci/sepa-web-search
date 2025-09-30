import asyncio
import nest_asyncio
from typing import List
from langchain_core.embeddings import Embeddings
from .inference import generate_embeddings_batch

# Allow nested event loops (needed for Jupyter/async contexts)
nest_asyncio.apply()


class GPUEmbeddingWrapper(Embeddings):
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
    
    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Synchronous wrapper for embedding documents - required by FAISS"""
        import logging
        logger = logging.getLogger(__name__)
        
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(generate_embeddings_batch(self.model_name, texts))
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    result = future.result()
                    return result
            else:
                return loop.run_until_complete(generate_embeddings_batch(self.model_name, texts))
        except RuntimeError as e:
            return asyncio.run(generate_embeddings_batch(self.model_name, texts))
    
    def embed_query(self, text: str) -> List[float]:
        embeddings = self.embed_documents([text])
        if not embeddings:
            raise Exception("Failed to generate embeddings - service may be unavailable")
        return embeddings[0]
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await generate_embeddings_batch(self.model_name, texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        embeddings = await self.aembed_documents([text])
        if not embeddings:
            raise Exception("Failed to generate embeddings - service may be unavailable")
        return embeddings[0]