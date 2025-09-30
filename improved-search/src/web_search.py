"""
Complete Web RAG Implementation for SePA
This file contains the full implementation extracted from the production system.
"""

import os
import asyncio
import aiohttp
from playwright.async_api import async_playwright
from readability import Document as ReadabilityDocument
import hashlib
import json
import pickle
import time
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

import torch
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from .embeddings import GPUEmbeddingWrapper
from sentence_transformers import CrossEncoder
from .inference import score_with_cross_encoder
import trafilatura
from openai import AsyncOpenAI
import PyPDF2
import io

from .config import Config
from .ranking import RecipRocalRankFusion

class RAG:
    """
    Complete RAG processor with all production features.
    This is the full implementation from the SePA system.
    """

    def __init__(self, embedding_model=None, cross_encoder_model=None, config=None):
        """Initialize the RAG processor with optimized parameters."""
        self.config = config or Config()

        # Initialize embedding model with GPU acceleration
        if embedding_model:
            self.embedding = embedding_model
        else:
            # Use GPU endpoints for production-level performance
            self.embedding = GPUEmbeddingWrapper(self.config.EMBEDDING_MODEL)
        self.embedding_model_name = self.config.EMBEDDING_MODEL

        # Initialize score tracking
        self._retrieval_scores = []

        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.COMPRESSION_CONFIG["chunk_size"],
            chunk_overlap=self.config.COMPRESSION_CONFIG["chunk_overlap"],
            length_function=len
        )

        # Semantic chunking parameters
        self.semantic_chunk_threshold = self.config.SEMANTIC_CHUNK_CONFIG["similarity_threshold"]

        # Search parameters
        self.search_params = {
            'num_results': self.config.SEARCH_CONFIG["num_results_per_query"],
            'timeout': 8,
            'max_content': 2500
        }

        # Cross-encoder model (uses GPU endpoints for reranking)
        self.cross_encoder_model_name = self.config.CROSS_ENCODER_MODEL
        # We'll use GPU endpoints directly via score_with_cross_encoder function

        # Initialize RRF
        self.rrf = RecipRocalRankFusion(
            k=self.config.RRF_CONFIG["k"],
            boost_factor=self.config.RRF_CONFIG["boost_factor"]
        )


    async def process_query(self, query, user_context, is_video_request=None, client=None):
        """
        Process the query asynchronously and build the vector store.

        This is the main entry point that orchestrates the entire pipeline.
        """
        self._retrieval_scores = []

        # Check if this is a video request
        if is_video_request is None:
            is_video_request = any(
                term in query.lower()
                for term in ["video", "watch", "show me", "demonstrate", "youtube"]
            )

        # Store raw search results for later reference
        raw_search_results = []

        # Phase 1: Generate multiple queries using LLM for comprehensive search
        search_queries = await self._generate_multi_queries(query, user_context, 4, client)

        # Phase 2: Execute searches in parallel
        all_result_lists = []

        if is_video_request:
            # For videos, search YouTube
            youtube_search_tasks = []
            for sq in search_queries:
                youtube_search_tasks.append(self._search_web(sq, trusted=False, youtube_only=True))

            youtube_result_lists = await asyncio.gather(*youtube_search_tasks, return_exceptions=True)

            valid_youtube_results = []
            for results in youtube_result_lists:
                if isinstance(results, list) and results:
                    valid_youtube_results.append(results)

            if valid_youtube_results:
                youtube_results = self.rrf.fuse(valid_youtube_results)
                raw_search_results.extend(youtube_results)

                # For videos, return special format
                video_metadata = self._extract_video_metadata_lightweight(youtube_results)
                if video_metadata:
                    return {'type': 'video_metadata', 'videos': video_metadata}
        else:
            # Regular search across all sources
            unified_search_tasks = []
            for sq in search_queries:
                unified_search_tasks.append(self._search_web(sq, trusted=False))

            search_results = await asyncio.gather(*unified_search_tasks, return_exceptions=True)

            for results in search_results:
                if isinstance(results, list) and results:
                    all_result_lists.append(results)

            if all_result_lists:
                # Phase 3: Apply RRF with trusted domain boosting
                merged_results = self.rrf.fuse(
                    all_result_lists,
                    trusted_domains=self.config.TRUSTED_DOMAINS
                )
                raw_search_results = merged_results

                # Phase 4: Process top results
                docs = await self._process_documents(merged_results[:10], query)

                if docs:
                    # Phase 5: Create vector store from documents
                    vector_store = FAISS.from_documents(
                        docs,
                        self.embedding,
                        normalize_L2=True
                    )
                    return vector_store

        # Fallback when no content is found
        fallback_content = (
            "No relevant content found for your query. "
            "Please try rephrasing or using different keywords."
        )
        fallback_doc = Document(
            page_content=fallback_content,
            metadata={"source": "fallback", "title": "No Results", "score": 0}
        )
        vector_store = FAISS.from_documents(
            [fallback_doc],
            self.embedding,
            normalize_L2=True
        )
        return vector_store

    async def _generate_multi_queries(self, original_query, user_context=None, max_queries=4, client=None):
        """Generate multiple diverse queries from the original query using LLM."""
        # Extract user context for query personalization
        context_parts = []
        if user_context:
            age = user_context.get('age')
            gender = user_context.get('gender')
            sport = user_context.get('sport')

            if gender:
                context_parts.append(f"{gender}")
            if sport:
                context_parts.append(f"{sport}")
            if age and age < 20:
                context_parts.append("teenage")
            elif age and 20 <= age < 30:
                context_parts.append("young adult")

        context_str = " ".join(context_parts) + " athletes" if context_parts else ""

        prompt = f"""Generate {max_queries} diverse, natural search queries for Google based on this question:

Original query: "{original_query}"

Create queries that will return high-quality, relevant results from the web. Make them natural and similar to how people actually search.

Guidelines:
1. First query: Keep it general and close to the original intent
2. Second query: Focus on scientific/evidence-based information (keep general)
3. Third query: Add natural demographic and health context if relevant: "{context_str}")
4. Fourth query: Focus on practical tips, best practices, or common mistakes

Important:
- Only ONE query (the third) should include demographic context, and only if it flows naturally
- Keep all queries concise and natural
- Use terminology that matches how content is indexed on the web

Return a JSON object with a "queries" array containing the generated queries."""

        try:
            if not client:
                client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

            response = await client.chat.completions.create(
                model=self.config.QUERY_EXPANSION_MODEL,
                messages=[
                    {"role": "system", "content": "You are a search query generator. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            response_json = json.loads(response.choices[0].message.content)
            generated_queries = response_json.get("queries", [])

            # Always include the original query
            all_queries = [original_query] + generated_queries[:max_queries-1]
            return all_queries

        except Exception as e:
            return [original_query]

    async def _search_web(self, query, trusted=True, include_video=False, youtube_only=False):
        """Search the web using Google Custom Search API."""
        async with aiohttp.ClientSession() as session:
            # Set up search parameters
            if youtube_only:
                site_query = " site:youtube.com"
                num_results = 3
            else:
                site_query = ""
                num_results = self.search_params['num_results']

            # Remove quotation marks from queries
            query = query.replace('"', '').replace("'", "")

            # Create the final search query
            full_query = f"{query}{site_query}"

            # Set up API parameters
            params = {
                'q': full_query,
                'key': os.getenv('GOOGLE_CSE_API_KEY'),
                'cx': os.getenv('GOOGLE_CSE_ENGINE_ID'),
                'num': num_results
            }

            # Execute the search
            async with session.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params
            ) as response:
                if response.status == 200:
                    results = await response.json()
                    items = results.get('items', [])
                    return items
                else:
                    return []

    def _reciprocal_rank_fusion(self, result_lists, k=60, return_scores=False,
                               trusted_domains=None, boost_factor=2.0):
        """
        Implement Reciprocal Rank Fusion (RRF) with trusted domain boosting.
        This is extracted directly from the original implementation.
        """

        # Use configured trusted domains if not provided
        if trusted_domains is None:
            trusted_domains = self.config.TRUSTED_DOMAINS

        # Dictionary to store RRF scores
        rrf_scores = {}

        # Calculate RRF score for each document
        for result_list in result_lists:
            for rank, result in enumerate(result_list):
                # Use URL as unique identifier
                doc_id = result.get('link', '')
                if not doc_id:
                    continue

                # Base RRF formula: 1 / (k + rank)
                score = 1 / (k + rank + 1)  # rank+1 because enumerate starts at 0

                # Check if this is a trusted domain
                is_trusted = any(domain in doc_id for domain in trusted_domains)
                if is_trusted:
                    score *= boost_factor

                if doc_id in rrf_scores:
                    rrf_scores[doc_id]['score'] += score
                    # Keep the result with more complete metadata
                    if len(result) > len(rrf_scores[doc_id]['result']):
                        rrf_scores[doc_id]['result'] = result
                else:
                    rrf_scores[doc_id] = {
                        'score': score,
                        'is_trusted': is_trusted,
                        'result': result
                    }

        # Sort by RRF score in descending order
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )

        # Log statistics
        if sorted_results:
            top_score = sorted_results[0][1]['score']
            top_10_trusted = sum(1 for item in sorted_results[:10] if item[1]['is_trusted'])
            # RRF merged results - top 10 contains trusted and other sources

        # Return results with or without scores
        if return_scores:
            return [(item[1]['result'], item[1]['score']) for item in sorted_results]
        else:
            return [item[1]['result'] for item in sorted_results]

    async def _process_documents(self, results, query):
        """Process search results with fetch-then-rerank pattern using cross-encoder."""

        docs = []

        # Maximum number of documents to process
        MAX_FETCH = self.config.SEARCH_CONFIG["max_fetch_candidates"]
        MAX_DOCS = self.config.SEARCH_CONFIG["max_final_documents"]

        # Stage 1: Fetch content for top N results
        fetch_candidates = []
        for i, result in enumerate(results[:MAX_FETCH]):
            title = result.get('title', '')
            link = result['link']
            is_youtube = 'youtube.com' in link or 'youtu.be' in link

            fetch_candidates.append({
                'link': link,
                'title': title,
                'snippet': result.get('snippet', ''),
                'is_youtube': is_youtube,
                'original_rank': i,
                'result': result
            })

        # Fetch content for all candidates IN PARALLEL
        fetched_results = await self._fetch_content_parallel(fetch_candidates, query)

        if not fetched_results:
            return []

        # Stage 2: Use cross-encoder to rerank based on actual content
        rerank_inputs = []
        for fetched in fetched_results:
            # Use the actual content preview for reranking
            text_representation = f"{fetched['candidate']['title']} {fetched['content_preview']}"
            rerank_inputs.append((query, text_representation))

        # Get cross-encoder scores using GPU endpoints
        try:
            scores = await score_with_cross_encoder(self.cross_encoder_model_name, rerank_inputs)

            # Apply scores to fetched results
            for i, fetched in enumerate(fetched_results):
                fetched['cross_encoder_score'] = float(scores[i])

            # Sort by cross-encoder score
            fetched_results = sorted(
                fetched_results,
                key=lambda x: x['cross_encoder_score'],
                reverse=True
            )
        except Exception as e:
            # Use fallback scoring based on original rank
            for i, fetched in enumerate(fetched_results):
                fetched['cross_encoder_score'] = 1.0 - (i * 0.1)

        # Stage 3: Apply relevance gate and create documents
        # Lower threshold for videos to ensure they're included
        relevance_threshold = 0.5

        for fetched in fetched_results[:MAX_DOCS]:
            # Lower threshold for videos or skip threshold check entirely
            if fetched.get('is_video'):
                # Videos always pass if they made it this far
                pass
            elif fetched['cross_encoder_score'] < relevance_threshold:
                continue

            # Create semantic chunks
            chunks = await self._semantic_chunk_text(
                fetched['full_content'],
                doc_title=fetched['candidate']['title']
            )

            # Create documents from chunks
            for chunk in chunks[:2]:  # Limit chunks per document
                if isinstance(chunk, dict):
                    content = chunk["content"]
                    metadata = chunk["metadata"]
                else:
                    content = chunk
                    metadata = {}

                metadata.update({
                    'source': fetched['candidate']['link'],
                    'title': fetched['candidate']['title'],
                    'score': fetched['cross_encoder_score']
                })

                doc = Document(
                    page_content=content[:self.search_params['max_content']],
                    metadata=metadata
                )
                docs.append(doc)
        return docs

    async def _fetch_content_parallel(self, fetch_candidates, query):
        """Fetch content from multiple URLs in parallel."""
        tasks = []

        for candidate in fetch_candidates:
            if candidate['is_youtube']:
                # For YouTube, format metadata without fetching
                tasks.append(self._format_youtube_content(candidate, query))
            else:
                # For regular URLs, fetch content
                tasks.append(self._fetch_content_with_candidate(candidate))

        # Execute all fetches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failed fetches
        fetched_results = []
        for result in results:
            if isinstance(result, dict) and result.get('full_content'):
                fetched_results.append(result)

        return fetched_results

    async def _fetch_content_with_candidate(self, candidate):
        """Fetch content for a single candidate."""
        try:
            content = await self._fetch_content(candidate['link'])
            if content:
                return {
                    'candidate': candidate,
                    'full_content': content,
                    'content_preview': content[:500]
                }
            else:
                # Fallback: Use snippet as content if fetch fails
                fallback_content = f"""Title: {candidate['title']}

{candidate['snippet']}

Note: Full content could not be fetched from {candidate['link']}. Using search snippet as fallback."""
                return {
                    'candidate': candidate,
                    'full_content': fallback_content,
                    'content_preview': candidate['snippet'],
                    'is_fallback': True
                }
        except Exception as e:
            # Last resort fallback
            fallback_content = f"""Title: {candidate['title']}

{candidate['snippet']}

Source: {candidate['link']}"""
            return {
                'candidate': candidate,
                'full_content': fallback_content,
                'content_preview': candidate['snippet'][:200],
                'is_fallback': True
            }

    async def _fetch_content(self, url):
        """Fetch and extract content from a URL using headless browser."""
        html = None

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)

                # Set better user agent and viewport
                context = await browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    viewport={'width': 1920, 'height': 1080}
                )
                page = await context.new_page()

                try:
                    # Use generous timeout for page to load
                    # For Reddit and dynamic sites, wait for networkidle
                    wait_until = 'networkidle' if 'reddit.com' in url else 'domcontentloaded'
                    response = await page.goto(url, timeout=20000, wait_until=wait_until)

                    if response is None:
                        return None

                    # Check content type for PDF handling
                    content_type = response.headers.get('content-type', '').lower()

                    if 'application/pdf' in content_type:
                        pdf_content = await response.body()
                        return self._parse_pdf(pdf_content)

                    # For HTML, let dynamic content load
                    # Extra wait for Reddit and dynamic sites
                    if 'reddit.com' in url:
                        await page.wait_for_timeout(3000)  # Wait longer for Reddit
                    else:
                        await page.wait_for_timeout(2000)  # Wait 2 seconds for JS rendering

                    html = await page.content()

                finally:
                    await context.close()
                    await browser.close()

            if html:
                return self._extract_text_from_html(html)

        except Exception as e:
            # Fallback to simple HTTP fetch if browser fails
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, timeout=15) as response:
                        if response.status == 200:
                            html = await response.text()
                            extracted = self._extract_text_from_html(html)
                            if extracted and len(extracted) > 100:  # Only return if meaningful content
                                return extracted
            except:
                pass

        return None

    def _extract_text_from_html(self, html):
        """Extract text from HTML using readability, then trafilatura, then BeautifulSoup."""
        # 1. Try Readability (Mozilla's Reader Mode algorithm) - often the best
        try:
            doc = ReadabilityDocument(html)
            title = doc.title()
            extracted_text = doc.summary(html_partial=True)  # gives html content
            # Clean the extracted HTML to get pure text
            soup = BeautifulSoup(extracted_text, 'html.parser')
            clean_text = ' '.join(soup.get_text(separator=' ', strip=True).split())
            if clean_text and len(clean_text) > 100:
                return clean_text
        except Exception:
            pass

        # 2. Try trafilatura (good for news articles)
        try:
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=True
            )
            if text and len(text) > 100:
                return text
        except Exception:
            pass

        # 3. Fallback to BeautifulSoup (basic but reliable)
        try:
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

            if text and len(text) > 100:
                return text
        except Exception as e:

            return None 

    def _parse_pdf(self, content):
        """Extract text from PDF content."""
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            text = ""
            for page_num in range(min(len(pdf_reader.pages), 10)):  # Limit to 10 pages
                page = pdf_reader.pages[page_num]
                text += page.extract_text()

            return text
        except Exception as e:
            return None

    async def _format_youtube_content(self, candidate, query):
        """Format YouTube video metadata without fetching."""
        # Create rich content for YouTube videos
        video_content = f"""YouTube Video: {candidate['title']}

Description: {candidate['snippet']}

This is a video resource about {query}. The video likely contains visual demonstrations and explanations relevant to the topic.

Video URL: {candidate['link']}"""

        return {
            'candidate': candidate,
            'full_content': video_content,
            'content_preview': candidate['snippet'],
            'is_video': True  # Mark as video for special handling
        }

    async def _semantic_chunk_text(self, text, doc_title="", max_chunk_size=1000):
        """
        Split text into semantic chunks based on sentence embeddings.
        This creates chunks at natural semantic boundaries.
        """
        try:
            # Split into sentences first
            sentences = re.split(r'(?<=[.!?])\s+', text)

            # Filter out very short sentences
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

            if len(sentences) < 2:
                # Too short for semantic chunking
                return [{"content": text, "metadata": {"section": "full"}}]

            # Get embeddings for all sentences (limit to 32 for batch size)
            embeddings = self.embedding.embed_documents(sentences[:32])
            embeddings = np.array(embeddings)

            # Calculate cosine similarity between consecutive sentences
            chunks = []
            current_chunk = [sentences[0]]
            current_chunk_size = len(sentences[0])

            for i in range(1, len(sentences)):
                # Calculate cosine similarity with previous sentence
                if i < len(embeddings):
                    cosine_sim = np.dot(embeddings[i-1], embeddings[i]) / (
                        np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
                    )
                else:
                    cosine_sim = 0.7

                # Check if we should start a new chunk
                should_split = (
                    cosine_sim < self.semantic_chunk_threshold or
                    current_chunk_size + len(sentences[i]) > max_chunk_size
                )

                if should_split and current_chunk:
                    # Save current chunk
                    chunk_content = ' '.join(current_chunk)

                    chunks.append({
                        "content": chunk_content,
                        "metadata": {
                            "section": self._identify_section_type(chunk_content),
                            "doc_title": doc_title,
                            "chunk_index": len(chunks)
                        }
                    })

                    # Start new chunk
                    current_chunk = [sentences[i]]
                    current_chunk_size = len(sentences[i])
                else:
                    # Add to current chunk
                    current_chunk.append(sentences[i])
                    current_chunk_size += len(sentences[i])

            # Don't forget the last chunk
            if current_chunk:
                chunk_content = ' '.join(current_chunk)
                chunks.append({
                    "content": chunk_content,
                    "metadata": {
                        "section": self._identify_section_type(chunk_content),
                        "doc_title": doc_title,
                        "chunk_index": len(chunks)
                    }
                })
            return chunks

        except Exception as e:
            # Fallback to simple chunking
            return [{"content": text[:max_chunk_size], "metadata": {"section": "fallback"}}]

    def _identify_section_type(self, text):
        """Identify the type of content in a chunk based on keywords."""
        text_lower = text.lower()

        # Define section patterns
        section_patterns = {
            "methodology": ["method", "study", "research", "participant", "protocol"],
            "results": ["result", "found", "showed", "demonstrated", "indicated"],
            "recommendations": ["recommend", "should", "suggest", "advised", "best practice"],
            "background": ["introduction", "background", "overview", "historically"],
            "practical": ["exercise", "training", "workout", "technique", "perform"],
            "recovery": ["recovery", "rest", "rehabilitation", "healing", "restoration"],
            "prevention": ["prevent", "avoid", "reduce risk", "injury prevention"],
            "metrics": ["measure", "metric", "vo2", "heart rate", "performance"]
        }

        # Count matches for each section type
        section_scores = {}
        for section, keywords in section_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                section_scores[section] = score

        # Return the section with highest score
        if section_scores:
            return max(section_scores.items(), key=lambda x: x[1])[0]
        return "general"

    def _extract_video_metadata_lightweight(self, results):
        """Extract video metadata without using ML models."""

        video_metadata = []
        MAX_VIDEOS = 6

        for i, result in enumerate(results[:MAX_VIDEOS]):
            title = result.get('title', 'Untitled Video')
            link = result.get('link', '')
            snippet = result.get('snippet', '')

            # Extract video ID
            video_id = None
            if 'youtube.com/watch?v=' in link:
                video_id = link.split('watch?v=')[1].split('&')[0]
            elif 'youtu.be/' in link:
                video_id = link.split('youtu.be/')[1].split('?')[0]

            video_info = {
                'title': title,
                'url': link,
                'description': snippet,
                'video_id': video_id,
                'rank': i + 1,
                'type': 'youtube_video'
            }

            video_metadata.append(video_info)

        return video_metadata
