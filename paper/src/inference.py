"""
HuggingFace Inference Endpoints for embeddings and reranking.
Also includes HTML/text processing utilities.
"""

import os
import asyncio
import logging
import concurrent.futures
import multiprocessing
from typing import List, Tuple, Optional
import httpx
import requests
import sys

logger = logging.getLogger(__name__)

# Process pool for CPU-intensive text processing
_process_pool = None
_process_pool_size = min(4, (multiprocessing.cpu_count() or 2))

# Load environment variables
HF_TOKEN = os.environ.get('HF_TOKEN')
HF_EMBEDDING_ENDPOINT = os.environ.get('HF_EMBEDDING_ENDPOINT')
HF_CROSSENCODER_ENDPOINT = os.environ.get('HF_CROSSENCODER_ENDPOINT')

# Check configuration
if not all([HF_TOKEN, HF_EMBEDDING_ENDPOINT, HF_CROSSENCODER_ENDPOINT]):
    raise ValueError(
        "HuggingFace Inference Endpoints required. Please set:\n"
        "  HF_TOKEN\n"
        "  HF_EMBEDDING_ENDPOINT\n"
        "  HF_CROSSENCODER_ENDPOINT"
    )


def get_process_pool() -> concurrent.futures.ProcessPoolExecutor:
    """Get or create the shared process pool for text processing"""
    global _process_pool
    if _process_pool is None:
        ctx = multiprocessing.get_context('spawn')
        _process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=_process_pool_size,
            mp_context=ctx
        )
    return _process_pool


async def generate_embeddings_batch(embedding_model_name: str, texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using HuggingFace Inference Endpoints.

    Args:
        embedding_model_name: Model name (for compatibility)
        texts: List of texts to embed

    Returns:
        List of embedding vectors
    """
    # Detect if we're in Jupyter/problematic async context
    if 'ipykernel' in sys.modules or 'nest_asyncio' in sys.modules:
        return _generate_embeddings_sync(texts)

    try:
        return await _generate_embeddings_async(texts)
    except (AssertionError, RuntimeError) as e:
        if "asyncio" in str(e) or "anyio" in str(e):
            return _generate_embeddings_sync(texts)
        raise


async def _generate_embeddings_async(texts: List[str]) -> List[List[float]]:
    """Async version of embedding generation."""
    # HuggingFace TEI has limits
    MAX_BATCH_SIZE = 32
    MAX_CHARS = 2048  # ~512 tokens

    # Truncate texts if needed
    truncated_texts = [text[:MAX_CHARS] for text in texts]

    # Process in batches
    all_embeddings = []

    for i in range(0, len(truncated_texts), MAX_BATCH_SIZE):
        batch = truncated_texts[i:i + MAX_BATCH_SIZE]

        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {"inputs": batch}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                HF_EMBEDDING_ENDPOINT,
                headers=headers,
                json=payload
            )

            if response.status_code == 200:
                embeddings = response.json()
                all_embeddings.extend(embeddings)
            elif response.status_code == 503:
                # Model loading - wait and retry once
                await asyncio.sleep(20)
                response = await client.post(
                    HF_EMBEDDING_ENDPOINT,
                    headers=headers,
                    json=payload
                )
                if response.status_code == 200:
                    embeddings = response.json()
                    all_embeddings.extend(embeddings)
                else:
                    raise Exception(f"Embedding API error after retry: {response.status_code}")
            else:
                raise Exception(f"Embedding API error {response.status_code}: {response.text}")

    return all_embeddings


def _generate_embeddings_sync(texts: List[str]) -> List[List[float]]:
    """Synchronous fallback for Jupyter environments."""
    MAX_BATCH_SIZE = 32
    MAX_CHARS = 2048

    truncated_texts = [text[:MAX_CHARS] for text in texts]
    all_embeddings = []

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    for i in range(0, len(truncated_texts), MAX_BATCH_SIZE):
        batch = truncated_texts[i:i + MAX_BATCH_SIZE]
        payload = {"inputs": batch}

        response = requests.post(
            HF_EMBEDDING_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            embeddings = response.json()
            all_embeddings.extend(embeddings)
        elif response.status_code == 503:
            # Model loading - wait and retry once
            import time
            time.sleep(20)
            response = requests.post(
                HF_EMBEDDING_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=30
            )
            if response.status_code == 200:
                embeddings = response.json()
                all_embeddings.extend(embeddings)
            else:
                raise Exception(f"Embedding API error after retry: {response.status_code}")
        else:
            raise Exception(f"Embedding API error {response.status_code}: {response.text}")

    return all_embeddings


async def score_with_cross_encoder(model_name: str, pairs: List[Tuple[str, str]]) -> List[float]:
    """
    Score query-document pairs using cross-encoder reranking.

    Args:
        model_name: Model name (for compatibility)
        pairs: List of (query, document) tuples

    Returns:
        List of relevance scores
    """
    # Detect if we're in Jupyter/problematic async context
    if 'ipykernel' in sys.modules or 'nest_asyncio' in sys.modules:
        return _score_pairs_sync(pairs)

    try:
        return await _score_pairs_async(pairs)
    except (AssertionError, RuntimeError) as e:
        if "asyncio" in str(e) or "anyio" in str(e):
            return _score_pairs_sync(pairs)
        raise


async def _score_pairs_async(pairs: List[Tuple[str, str]]) -> List[float]:
    """Async version of cross-encoder scoring."""
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    scores = []

    # Group pairs by query for efficient processing
    from collections import defaultdict
    query_groups = defaultdict(list)
    pair_indices = {}

    for idx, (query, text) in enumerate(pairs):
        query_groups[query].append(text)
        pair_indices[(query, text)] = idx

    async with httpx.AsyncClient(timeout=30.0) as client:
        for query, texts in query_groups.items():
            # Try flat structure first
            payload = {
                "query": query,
                "texts": texts[:3]  # Limit for API constraints
            }

            response = await client.post(
                HF_CROSSENCODER_ENDPOINT,
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                # Try nested format as fallback
                payload = {"inputs": {"query": query, "texts": texts[:3]}}
                response = await client.post(
                    HF_CROSSENCODER_ENDPOINT,
                    headers=headers,
                    json=payload
                )

            if response.status_code == 200:
                result = response.json()
                # Handle different response formats
                if isinstance(result, list):
                    if isinstance(result[0], dict) and 'score' in result[0]:
                        score_values = [item['score'] for item in result]
                    else:
                        score_values = result

                    # Map scores back to original pairs
                    texts_scored = texts[:3] if len(texts) > 3 else texts
                    for text, score in zip(texts_scored, score_values):
                        if (query, text) in pair_indices:
                            idx = pair_indices[(query, text)]
                            scores.append((idx, float(score)))

                    # Add default scores for texts we couldn't score
                    if len(texts) > 3:
                        for text in texts[3:]:
                            if (query, text) in pair_indices:
                                idx = pair_indices[(query, text)]
                                scores.append((idx, 0.0))
            else:
                raise Exception(f"Reranker API error {response.status_code}: {response.text}")

    # Sort by original index and return scores
    scores.sort(key=lambda x: x[0])
    return [score for _, score in scores]


def _score_pairs_sync(pairs: List[Tuple[str, str]]) -> List[float]:
    """Synchronous fallback for Jupyter environments."""
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    scores = []

    # Group pairs by query
    from collections import defaultdict
    query_groups = defaultdict(list)
    pair_indices = {}

    for idx, (query, text) in enumerate(pairs):
        query_groups[query].append(text)
        pair_indices[(query, text)] = idx

    for query, texts in query_groups.items():
        # Try flat structure
        payload = {
            "query": query,
            "texts": texts[:3]
        }

        response = requests.post(
            HF_CROSSENCODER_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            # Try nested format
            payload = {"inputs": {"query": query, "texts": texts[:3]}}
            response = requests.post(
                HF_CROSSENCODER_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=30
            )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list):
                if isinstance(result[0], dict) and 'score' in result[0]:
                    score_values = [item['score'] for item in result]
                else:
                    score_values = result

                # Map scores back
                texts_scored = texts[:3] if len(texts) > 3 else texts
                for text, score in zip(texts_scored, score_values):
                    if (query, text) in pair_indices:
                        idx = pair_indices[(query, text)]
                        scores.append((idx, float(score)))

                # Default scores for unscored texts
                if len(texts) > 3:
                    for text in texts[3:]:
                        if (query, text) in pair_indices:
                            idx = pair_indices[(query, text)]
                            scores.append((idx, 0.0))
        elif response.status_code == 503:
            # Model loading - wait and retry once
            import time
            time.sleep(20)
            response = requests.post(
                HF_CROSSENCODER_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list):
                    score_values = result if isinstance(result[0], float) else [r['score'] for r in result]
                    texts_scored = texts[:3] if len(texts) > 3 else texts
                    for text, score in zip(texts_scored, score_values):
                        if (query, text) in pair_indices:
                            idx = pair_indices[(query, text)]
                            scores.append((idx, float(score)))
        else:
            raise Exception(f"Reranker API error: {response.status_code}")

    # Sort by original index and return scores
    scores.sort(key=lambda x: x[0])
    return [score for _, score in scores]


# HTML/Text Processing Functions (Essential - NOT cold start related!)

async def extract_text_from_html(html: str) -> Optional[str]:
    """Extract text from HTML using trafilatura in process pool"""
    loop = asyncio.get_event_loop()
    pool = get_process_pool()

    # Run in process pool
    text = await loop.run_in_executor(pool, _extract_text_in_process, html)
    return text


def _extract_text_in_process(html: str) -> Optional[str]:
    """Extract text in a separate process (must be picklable)"""
    try:
        import trafilatura
        return trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=True
        )
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return None


async def parse_html_with_beautifulsoup(html: str) -> Optional[str]:
    """Parse HTML with BeautifulSoup in process pool"""
    loop = asyncio.get_event_loop()
    pool = get_process_pool()

    # Run in process pool
    text = await loop.run_in_executor(pool, _parse_html_in_process, html)
    return text


def _parse_html_in_process(html: str) -> Optional[str]:
    """Parse HTML in a separate process (must be picklable)"""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text
    except Exception as e:
        logger.error(f"Error parsing HTML: {e}")
        return None


async def chunk_text_semantically(text: str, max_chunk_size: int = 1000) -> List[str]:
    """Chunk text semantically using NLP in process pool"""
    loop = asyncio.get_event_loop()
    pool = get_process_pool()

    # Run in process pool
    chunks = await loop.run_in_executor(pool, _chunk_text_in_process, text, max_chunk_size)
    return chunks


def _chunk_text_in_process(text: str, max_chunk_size: int) -> List[str]:
    """Chunk text in a separate process (must be picklable)"""
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        # Ensure chunks are small enough for embedding model
        safe_chunk_size = min(max_chunk_size, 1500)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=safe_chunk_size,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_text(text)

        # Additional safety check - ensure no chunk exceeds 2048 chars
        MAX_CHARS = 2048
        safe_chunks = []
        for chunk in chunks:
            if len(chunk) > MAX_CHARS:
                # Split oversized chunks
                for i in range(0, len(chunk), MAX_CHARS):
                    safe_chunks.append(chunk[i:i + MAX_CHARS])
            else:
                safe_chunks.append(chunk)

        return safe_chunks
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        # Fallback to simple chunking
        MAX_CHARS = 2048
        chunks = []
        for i in range(0, len(text), MAX_CHARS):
            chunks.append(text[i:i + MAX_CHARS])
        return chunks


def cleanup_process_pool():
    """Cleanup the process pool on shutdown"""
    global _process_pool
    if _process_pool is not None:
        _process_pool.shutdown(wait=True)
        _process_pool = None


# Register cleanup on module unload
import atexit
atexit.register(cleanup_process_pool)