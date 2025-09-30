import os
import asyncio

import yaml
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI

from cache import RAGCache
from models import load_embedding_model, load_cross_encoder, get_openai_client
from search import search_web, process_documents

class RAG:

    def __init__(self, embedding_model=None, cross_encoder_model=None, async_openai_client=None):
        """

        Args:
            embedding_model: Pre-loaded embedding model
            cross_encoder_model: Pre-loaded cross encoder model
            async_openai_client: Initialized AsyncOpenAI client
        """
        # Load configuration
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)

        # Use provided models or load new ones
        self.embedding = embedding_model or load_embedding_model()
        self.cross_encoder = cross_encoder_model or load_cross_encoder()
        self.async_openai_client = async_openai_client or get_openai_client()


        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['search']['chunk_size'],
            chunk_overlap=self.config['search']['chunk_overlap'],
            length_function=len
        )

        self.search_params = {
            'num_results_trusted': self.config['search']['num_results_trusted'],
            'num_results_general': self.config['search']['num_results_general'],
            'num_results_youtube': self.config['search']['num_results_youtube'],
            'timeout_seconds': self.config['search']['timeout_seconds'],
            'max_content_length': self.config['search']['max_content_length']
        }

        # Initialize cache system
        cache_config = self.config['cache']
        self.cache = RAGCache(
            base_dir=cache_config['base_dir'],
            ttl_days=cache_config['ttl_days'],
            memory_size=cache_config['memory_size'],
            similarity_threshold=cache_config['similarity_threshold']
        )
        self.cache.set_embedding_model(self.embedding)

        self._original_search_web = self._search_web_direct
        self._search_web = self._cached_search_web

    async def cleanup_cache(self):
        return await self.cache.cleanup_expired()

    async def _cached_search_web(self, query, trusted=True, include_video=False, youtube_only=False):
        if youtube_only:
            return await self._original_search_web(query, trusted, include_video, youtube_only)

        normalized_query = self.cache._normalize_query(query)

        cache_result = await self.cache.get(normalized_query, trusted, include_video)
        if cache_result is not None:
            return cache_result

        results = await self._original_search_web(query, trusted, include_video, youtube_only)

        if results:
            await self.cache.put(normalized_query, results, trusted, include_video)

        return results

    async def _search_web_direct(self, query, trusted=True, include_video=False, youtube_only=False):
        return await search_web(
            query,
            self.embedding,
            self.cross_encoder,
            trusted,
            include_video,
            youtube_only
        )

    async def process_query(self, query, user_context, is_video_request=None):
        """Process the query asynchronously and build the vector store.

        Args:
            query: The user's query
            user_context: Dictionary with user context information
            is_video_request: Boolean indicating if this is a video request

        Returns:
            FAISS vector store with processed documents
        """
        if is_video_request is None:
            video_keywords = self.config['youtube']['filter_patterns']
            is_video_request = any(term in query.lower() for term in ["video", "watch", "show me", "demonstrate", "youtube"])

        raw_search_results = []

        primary_query = query

        if is_video_request:
            youtube_results = await self._search_web(query, trusted=False, youtube_only=True)

            if youtube_results:
                raw_search_results.extend(youtube_results)

                docs = await process_documents(
                    youtube_results,
                    query,
                    self.embedding,
                    self.cross_encoder,
                    self.text_splitter,
                    self.search_params
                )

                if docs:
                    vector_store = FAISS.from_documents(docs, self.embedding, normalize_L2=True)
                    return vector_store
        else:
            search_results = await self._search_web(primary_query, trusted=True)

            if search_results and len(search_results) > 0:
                raw_search_results.extend(search_results)

                docs = await process_documents(
                    search_results,
                    query,
                    self.embedding,
                    self.cross_encoder,
                    self.text_splitter,
                    self.search_params
                )

                if docs:
                    vector_store = FAISS.from_documents(docs, self.embedding, normalize_L2=True)
                    return vector_store

            general_results = await self._search_web(primary_query, trusted=False)

            if general_results and len(general_results) > 0:
                raw_search_results.extend(general_results)

                from search import fetch_content
                first_result = general_results[0]
                content = await fetch_content(first_result['link'])

                if content:
                    doc = Document(
                        page_content=content[:self.search_params['max_content_length']],
                        metadata={
                            'source': first_result['link'],
                            'title': first_result.get('title', 'General Search Result'),
                            'score': 0
                        }
                    )
                    vector_store = FAISS.from_documents([doc], self.embedding, normalize_L2=True)
                    return vector_store

        fallback_content = (
            "No trusted content found on high-quality sources. "
            "Consider using general exercise physiology knowledge."
        )
        fallback_doc = Document(
            page_content=fallback_content,
            metadata={"source": "fallback", "title": "Fallback", "score": 0}
        )
        vector_store = FAISS.from_documents([fallback_doc], self.embedding, normalize_L2=True)
        return vector_store

    async def rag_generate_response(self, user_input, user_context, conversation_history,
                                  vector_store=None):
        """Generate response using the vector store.

        Args:
            user_input: The user's query
            user_context: Dictionary with user context information
            conversation_history: List of previous conversation turns
            vector_store: Pre-built FAISS vector store from process_query

        Returns:
            Dictionary with answer, sources, and metadata
        """
        if vector_store is None:
            return {
                "answer": "No search results available to generate a response.",
                "sources": [],
                "has_video": False,
                "query": user_input,
                "source_documents": [],
                "raw_results": []
            }

        try:
            # Build context from conversation history
            conversation_context = ""
            if conversation_history:
                recent_history = conversation_history[-3:]  # Last 3 turns
                conversation_context = "\n".join([
                    f"Previous: {turn.get('user', '')}\nResponse: {turn.get('assistant', '')}"
                    for turn in recent_history
                ])

            # Create context-aware prompt
            prompt_template = f"""You are a knowledgeable sports science and athletic health expert. Based on the provided sources, answer the user's question with evidence-based information.

User Context: Age {user_context.get('age', 'unknown')}, Gender {user_context.get('gender', 'unknown')}, Sport {user_context.get('sport', 'unknown')}

Previous conversation context:
{conversation_context}

Question: {{question}}

Context from sources:
{{context}}

Provide a helpful, accurate response. If the context includes YouTube videos, mention them as recommended resources. Always cite your sources.

Answer:"""

            prompt = PromptTemplate(
                input_variables=["question", "context"],
                template=prompt_template
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    model=self.config['models']['llm_model'],
                    temperature=self.config['response']['temperature'],
                    max_tokens=self.config['response']['max_tokens'],
                    openai_api_key=os.environ.get("OPENAI_API_KEY")
                ),
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": self.config['response']['retrieval_k']}
                ),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )

            result = qa_chain({"query": user_input})
            answer = result['result']
            docs = result['source_documents']

            sources = []
            has_video = False

            for doc in docs:
                source_url = doc.metadata.get('source', '')
                if source_url not in sources:
                    sources.append(source_url)

                if 'youtube.com' in source_url or 'youtu.be' in source_url:
                    has_video = True

            # Return the result with sources
            return {
                "answer": answer,
                "sources": list(set(sources)),
                "has_video": has_video,
                "query": user_input,
                "source_documents": docs,
                "raw_results": getattr(self, 'raw_search_results', [])
            }

        except Exception as e:
            return {
                "answer": "I'm sorry, but I encountered an error processing your request. Please try again.",
                "sources": [],
                "has_video": False,
                "query": user_input,
                "source_documents": [],
                "raw_results": []
            }


def create_rag():
    """Create a RAG instance with pre-loaded models for efficiency"""
    embedding_model = load_embedding_model()
    cross_encoder_model = load_cross_encoder()
    async_openai_client = get_openai_client()

    return RAG(embedding_model, cross_encoder_model, async_openai_client)