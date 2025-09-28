import os
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from openai import AsyncOpenAI


def load_embedding_model():
    """Load the embedding model for semantic search"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu', 'trust_remote_code': True},
    )


def load_cross_encoder():
    """Load the cross-encoder model for re-ranking"""
    return CrossEncoder("cross-encoder/ms-marco-electra-base", device='cpu')


def get_openai_client():
    """Initialize async OpenAI client"""
    return AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )