"""
SePA Web Search Agent - Athletic Performance RAG System
"""

from .web_search import RAG
from .config import Config
from .ranking import RecipRocalRankFusion
from .response import ResponseGenerator
from .embeddings import GPUEmbeddingWrapper
from .inference import generate_embeddings_batch, score_with_cross_encoder

__version__ = "1.0.0"
__author__ = "SePA Research Team"

__all__ = [
    "RAG",
    "Config",
    "RecipRocalRankFusion",
    "ResponseGenerator",
    "GPUEmbeddingWrapper",
    "generate_embeddings_batch",
    "score_with_cross_encoder"
]
