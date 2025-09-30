"""
Response Generation and Context Compression Module
Extracted from the original webRag.py implementation
"""

import json
import re
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document
from openai import OpenAI
import os 

class ResponseGenerator:
    """Handles context compression and response generation."""

    def __init__(self, config):
        self.config = config
        # Use synchronous client to avoid nest_asyncio conflicts
        self.llm_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    async def compress_context(
        self,
        docs: List[Document],
        user_query: str,
        user_context: Optional[Dict] = None,
        numbered_sources: Optional[List] = None,
        client: Optional[OpenAI] = None,
        model: str = "gpt-4o"
    ) -> Optional[str]:
        """
        Compress retrieved documents into a concise context with citations.
        This is a critical innovation that reduces token usage while preserving information.
        """
        if not docs:
            return None

        try:
            # Create source mapping for citations
            source_map = {}
            unique_sources = []
            source_titles = []

            # Extract unique sources from docs
            for doc in docs:
                source = doc.metadata.get("source")
                title = doc.metadata.get("title", "Unknown Title")
                if source and source != "fallback" and source not in unique_sources:
                    unique_sources.append(source)
                    source_titles.append(title)

            # Create source index mapping
            for i, source in enumerate(unique_sources):
                source_map[source] = i + 1

            # Format documents with proper numbered citations
            formatted_docs = []
            for doc in docs[:5]:  # Limit to top 5 docs
                source_num = source_map.get(doc.metadata.get('source'), 0)
                if source_num > 0:
                    doc_text = f"[From Source {source_num}: {doc.metadata.get('title', 'Unknown')}]\n"

                    # Add section information if available
                    if doc.metadata.get('section'):
                        doc_text += f"Section: {doc.metadata['section']}\n"

                    doc_text += f"{doc.page_content}\n"
                    formatted_docs.append(doc_text)

            # Combine all documents
            full_context = "\n---\n".join(formatted_docs)

            # Create numbered source list for the prompt
            numbered_sources_text = ""
            for i, (source, title) in enumerate(zip(unique_sources, source_titles)):
                numbered_sources_text += f"[{i+1}] {title}\n"

            # Build user context description
            context_desc = "general athlete"
            if user_context:
                parts = []
                if user_context.get('age'):
                    parts.append(f"{user_context['age']}-year-old")
                if user_context.get('gender'):
                    parts.append(user_context['gender'])
                if user_context.get('sport'):
                    parts.append(f"{user_context['sport']} athlete")
                if parts:
                    context_desc = " ".join(parts)

            # Create compression prompt with inline citation instructions
            compression_prompt = f"""You are a health information specialist. Your task is to extract and synthesize the most relevant information from multiple sources to answer a specific query.

User Query: "{user_query}"

User Context: {context_desc}

Sources Available for Citation:
{numbered_sources_text}

Retrieved Information (with source indicators):
{full_context}

Instructions:
1. Extract ONLY the key facts, data points, and recommendations that directly relate to answering the user's query
2. Remove redundant information if multiple sources say the same thing
3. Preserve specific numbers, percentages, and scientific findings
4. CRITICAL: For every fact you extract, you MUST include its source citation using [1], [2], etc. based on the source numbers above
5. Organize the information logically by theme or relevance
6. Keep the compressed version concise but comprehensive - in under roughly {self.config.COMPRESSION_CONFIG['target_tokens']} tokens.

Example format:
- VO2 max can be improved by 10-15% through HIIT training over 8 weeks [1]
- Recovery periods of 48-72 hours are recommended between intense sessions [2]
- Combining aerobic and anaerobic training yields best results for athletes [1][3]

Output the compressed information with numbered citations [1], [2], etc. for all factual claims."""

            # Use provided client or create new one
            if not client:
                client = self.llm_client

            # Use sync client to avoid async conflicts
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting and summarizing health and athletic performance information. Always preserve source citations."
                    },
                    {"role": "user", "content": compression_prompt}
                ],
                max_tokens=self.config.COMPRESSION_CONFIG['target_tokens'] + 250  # Allow some buffer
            )

            compressed = response.choices[0].message.content.strip()

            return compressed

        except Exception as e:
            return None

    async def rag_generate_response(
        self,
        user_input: str,
        user_context: Dict,
        conversation_history: List,
        verbosity_level: str = "moderate",
        vector_store=None,
        system_prompt: Optional[str] = None,
        client: Optional[OpenAI] = None,
        model: str = "gpt-4o"
    ) -> Dict[str, Any]:
        """
        Generate a response using the RAG system.

        This is the final step that creates the actual response to the user.
        """
        # Convert verbosity level to configuration
        if isinstance(verbosity_level, str):
            verbosity_map = {"low": 1, "moderate": 2, "high": 3}
            verbosity_level = verbosity_map.get(verbosity_level.lower(), 2)

        # Map numeric levels to max_tokens
        max_tokens_map = {
            1: self.config.RESPONSE_CONFIG["verbosity_levels"]["low"]["max_tokens"],
            2: self.config.RESPONSE_CONFIG["verbosity_levels"]["moderate"]["max_tokens"],
            3: self.config.RESPONSE_CONFIG["verbosity_levels"]["high"]["max_tokens"]
        }
        max_tokens = max_tokens_map.get(verbosity_level, 400)

        if not vector_store:
            return {
                "answer": "I couldn't find relevant information. Please try a different question.",
                "sources": [],
                "raw_results": [],
                "retrieval_scores": []
            }

        # Check for simple acknowledgments
        simple_responses = ["thank you", "thanks", "ok", "okay", "got it", "understood", "great"]
        if user_input.lower().strip() in simple_responses:
            return {
                "answer": "You're welcome! Let me know if you have any other questions.",
                "sources": [],
                "raw_results": [],
                "retrieval_scores": []
            }

        # Check if this is a video request
        is_video_request = any(
            term in user_input.lower()
            for term in ["video", "watch", "show me", "demonstrate", "youtube"]
        )

        # Handle special video metadata format
        if isinstance(vector_store, dict) and vector_store.get('type') == 'video_metadata':
            return await self._generate_video_response(
                user_input,
                vector_store['videos'],
                user_context,
                conversation_history,
                verbosity_level,
                max_tokens,
                client,
                model
            )

        # Retrieve documents using the vector store
        retriever = vector_store.as_retriever(
            search_kwargs={"k": self.config.SEARCH_CONFIG["retrieval_k"]}
        )
        docs = retriever.get_relevant_documents(user_input)

        # Track retrieval scores
        retrieval_scores = []
        for doc in docs:
            if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                retrieval_scores.append(float(doc.metadata['score']))

        # Check for fallback state
        if len(docs) == 1 and docs[0].metadata.get("source") == "no_relevant_results":
            return {
                "answer": "Based on the provided sources, I could not find specific information to answer your question.",
                "sources": [],
                "has_video": False,
                "query": user_input,
                "source_documents": [],
                "raw_results": [],
                "retrieval_scores": []
            }

        # Extract unique sources for citations
        unique_sources = []
        source_titles = []

        for doc in docs:
            source = doc.metadata.get("source")
            title = doc.metadata.get("title", "Unknown Title")
            if source and source != "fallback" and source not in unique_sources:
                unique_sources.append(source)
                source_titles.append(title)

        # Apply context compression
        numbered_sources = list(zip(unique_sources, source_titles))
        compressed_context = await self.compress_context(
            docs, user_input, user_context, numbered_sources, client, model
        )

        # Use compressed context if available
        if compressed_context:
            context = compressed_context
        else:
            # Fallback to regular context
            context = "\n\n".join([doc.page_content for doc in docs[:3]])

        # Build user description
        context_info = []
        if user_context.get('age') and user_context.get('gender'):
            context_info.append(f"{user_context.get('age')}-year-old {user_context.get('gender')}")
        if user_context.get('sport'):
            context_info.append(f"athlete in {user_context.get('sport')}")

        # Create verbosity-specific instructions
        verbosity_instructions = {
            1: """RESPONSE STYLE: SIMPLE (HARD LIMIT)
            - Your response MUST be extremely brief (1-2 sentences, max 100 tokens)
            - Provide only the most essential conclusion or insight
            - No detailed explanations or context""",

            2: """RESPONSE STYLE: MODERATE (HARD LIMIT)
            - Your response MUST be informative but balanced (3-6 sentences, max 400 tokens)
            - Include key insights and practical implications
            - Draw clear conclusions from any data mentioned
            - Provide actionable recommendations when relevant""",

            3: """RESPONSE STYLE: COMPREHENSIVE (HARD LIMIT)
            - Your response should be thorough and analytical (6-12 sentences, max 800 tokens)
            - Provide detailed interpretation of all data, statistics, and results
            - Explain the significance and implications of findings
            - Draw meaningful conclusions while maintaining proper causal inference principles
            - Discuss potential explanations, associations, and recommendations
            - Connect findings to broader health and performance patterns
            - Use appropriate language that distinguishes correlation from causation"""
        }

        verbosity_instruction = verbosity_instructions.get(
            verbosity_level,
            verbosity_instructions[2]
        )

        # Parse health data from conversation history if available
        health_data_summary = self._extract_health_context(conversation_history)

        # Define the system message
        base_system_message = f"""
        You are an information synthesis assistant. Your task is to answer the user's query based ONLY on the information provided in the 'Retrieved Information' section below.

        CRITICAL INSTRUCTIONS:
        - DO NOT use any external knowledge or information not present in the Retrieved Information
        - Your entire response must be derived from the provided text
        - If the 'Retrieved Information' does not contain an answer to the user's query, you MUST explicitly state that
        - Do not create a 'References' section in your output. Only use the inline bracketed citations
        - If sources contradict each other, acknowledge this and cite both perspectives

        RESPONSE GUIDELINES:
        - Be conversational and helpful within the constraints of the provided information
        - When you cite information using [1], [2], etc., these correspond to the sources listed
        - Focus on synthesizing the retrieved information to answer the specific query

        The user is a {' '.join(context_info) if context_info else 'collegiate athlete'}.

        {health_data_summary}

        Verbosity level for your answer: {verbosity_instruction}
        """

        # Combine with custom system prompt if provided
        system_message = f"{base_system_message}\n\n{system_prompt}" if system_prompt else base_system_message

        # Add video-specific instructions if needed
        if is_video_request:
            system_message += """
            IF THERE ARE YOUTUBE/VIDEO REFERENCES IN THE RETRIEVED INFORMATION:
            - Evaluate each video's relevance based on its title and description
            - ALWAYS include video URLs in your response when videos are provided
            - Format video references as: "Here are some helpful videos: <title> - <url>"
            """

        # Create source context for the user message
        source_context = "\n\nSources Available:\n"
        for i, (source, title) in enumerate(zip(unique_sources, source_titles)):
            source_context += f"[{i+1}] {title}\n"

        # Define the user message
        user_message = f"""Use the provided information to answer the query.

{source_context}

Retrieved Information:
{context}

User is asking this:
{user_input}"""

        # Generate the response
        try:
            if not client:
                client = self.llm_client

            # Clean conversation history for RAG
            clean_history = self._clean_conversation_history(conversation_history)

            # Create messages for the LLM
            messages = [
                {"role": "system", "content": system_message}
            ]

            # Add recent conversation history
            for msg in clean_history[-2:]:  # Last 2 exchanges
                messages.append(msg)

            # Add current query
            messages.append({"role": "user", "content": user_message})

            # Use sync client to avoid async conflicts
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )

            answer = response.choices[0].message.content

            # Format sources for response
            sources = []
            # Found unique sources to format
            for i, (source, title) in enumerate(zip(unique_sources, source_titles)):
                sources.append({
                    'title': title,
                    'url': source,
                    'relevance_score': retrieval_scores[i] if i < len(retrieval_scores) else 0.0
                })
                # Source added to list

            # Returning sources in result dictionary
            result = {
                "answer": answer,
                "sources": sources,
                "has_video": is_video_request,
                "query": user_input,
                "source_documents": docs,
                "raw_results": [],
                "retrieval_scores": retrieval_scores
            }
            # Result prepared with sources
            return result

        except Exception as e:
            
            return {
                "answer": f"I encountered an error generating the response: {str(e)}",
                "sources": [],
                "has_video": False,
                "query": user_input,
                "source_documents": [],
                "raw_results": [],
                "retrieval_scores": []
            }

    async def _generate_video_response(
        self,
        user_input: str,
        videos: List[Dict],
        user_context: Dict,
        conversation_history: List,
        verbosity_level: int,
        max_tokens: int,
        client: Optional[OpenAI],
        model: str
    ) -> Dict[str, Any]:
        """Generate response for video-specific queries."""

        # Create video context
        video_context = "AVAILABLE YOUTUBE VIDEOS:\n\n"
        for i, video in enumerate(videos, 1):
            video_context += f"{i}. **{video['title']}**\n"
            video_context += f"   URL: {video['url']}\n"
            if video['description']:
                video_context += f"   Description: {video['description']}\n"
            video_context += "\n"

        # Build user context description
        context_info = []
        if user_context.get('age') and user_context.get('gender'):
            context_info.append(f"{user_context.get('age')}-year-old {user_context.get('gender')}")
        if user_context.get('sport'):
            context_info.append(f"athlete in {user_context.get('sport')}")

        # Create system prompt
        system_message = f"""You are SePA, a specialized health and sports performance assistant for student athletes.

CONTEXT: You are helping a {' '.join(context_info) if context_info else 'student athlete'}.

IMPORTANT INSTRUCTIONS FOR VIDEO REQUESTS:
- Evaluate each YouTube video's relevance based on its title and description
- Recommend the most relevant videos that match the user's specific request
- Explain why each recommended video is relevant
- Be honest if none of the videos are highly relevant
- Focus on practical, actionable video content

Available videos are provided below. Analyze them and provide recommendations based on the user's query."""

        # Create messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"USER QUERY: {user_input}\n\n{video_context}"}
        ]

        try:
            if not client:
                client = self.llm_client

            # Use sync client to avoid async conflicts
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens
            )

            answer = response.choices[0].message.content

            # Create sources from videos
            sources = []
            for video in videos:
                sources.append({
                    'title': video['title'],
                    'url': video['url'],
                    'relevance_score': 1.0
                })

            return {
                "answer": answer,
                "sources": sources,
                "has_video": True,
                "query": user_input,
                "source_documents": videos,
                "raw_results": videos,
                "retrieval_scores": [1.0] * len(videos),
                "processing_method": "lightweight_video_processing"
            }

        except Exception as e:
            return {
                "answer": f"I found {len(videos)} relevant videos but encountered an error. Please try again.",
                "sources": [],
                "has_video": True,
                "query": user_input,
                "source_documents": videos,
                "raw_results": videos,
                "retrieval_scores": []
            }

    def _extract_health_context(self, conversation_history: List) -> str:
        """Extract health context from conversation history."""
        # This would parse health data from the conversation
        return ""

    def _clean_conversation_history(self, conversation_history: List) -> List[Dict]:
        """Clean conversation history for LLM consumption."""
        clean_history = []
        for msg in conversation_history:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
                if role in ["user", "assistant"] and content and "tool_calls" not in msg:
                    clean_history.append({"role": role, "content": content})
        return clean_history
