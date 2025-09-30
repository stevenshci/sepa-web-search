"""
Configuration settings for the SePA Web Search Agent
"""

class Config:
    """Configuration class for RAG system parameters"""

    # Trusted domains for athletic and health information
    TRUSTED_DOMAINS = [
        # Sports medicine organizations
        "sportsmed.org",
        "bjsm.bmj.com",
        "sportsmedicine.stanford.edu",
        "hopkinsmedicine.org",
        "med.stanford.edu/sports",
        "mayoclinic.org",
        "health.harvard.edu",
        "my.clevelandclinic.org",

        # Athletic performance resources
        "acsm.org",  # American College of Sports Medicine
        "jospt.org",  # Journal of Orthopaedic & Sports Physical Therapy
        "gssiweb.org",  # Gatorade Sports Science Institute
        "acewebcontent.azureedge.net",  # ACE Fitness
        "hopkinsmedicine.org/sports-medicine",
        "hss.edu",  # Hospital for Special Surgery
        "stopsportsinjuries.org",
        "teamusa.org",
        "ncaa.org/sports-science",
        "strengthandconditioning.org",
        "scienceforsport.com",
        "sportscienceinsider.com",
        "medlineplus.gov",

        # Research databases
        "pubmed.ncbi.nlm.nih.gov",
        "journals.humankinetics.com"
    ]

    # Model specifications
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"
    QUERY_EXPANSION_MODEL = "gpt-4o-mini"

    # Search parameters
    SEARCH_CONFIG = {
        "num_queries": 4,  # Number of expanded queries
        "num_results_per_query": 8,  # Results per search query
        "max_fetch_candidates": 10,  # URLs to fetch content from
        "max_final_documents": 4,  # Documents after reranking
        "retrieval_k": 5,  # Documents to retrieve from vector store
    }

    # RRF parameters
    RRF_CONFIG = {
        "k": 60,  # RRF constant (standard value from literature)
        "boost_factor": 2.0,  # Multiplier for trusted domain scores
    }


    # Context compression parameters
    COMPRESSION_CONFIG = {
        "target_tokens": 1000,  # Target size for compressed context
        "preserve_citations": True,  # Always maintain source citations
        "chunk_size": 800,  # Size of text chunks
        "chunk_overlap": 150,  # Overlap between chunks
        "max_chunk_chars": 1500,  # Maximum characters per chunk (conservative)
    }

    # Semantic chunking parameters
    SEMANTIC_CHUNK_CONFIG = {
        "similarity_threshold": 0.5,  # Cosine similarity threshold for chunk boundaries
        "max_chunk_size": 1000,  # Maximum chunk size in characters
        "min_sentence_length": 20,  # Minimum sentence length to keep
        "max_sentence_length": 2000,  # Maximum sentence length before splitting
    }

    # Response generation parameters
    RESPONSE_CONFIG = {
        "verbosity_levels": {
            "low": {"max_tokens": 100, "sentences": "1-2"},
            "moderate": {"max_tokens": 400, "sentences": "3-6"},
            "high": {"max_tokens": 800, "sentences": "6-12"}
        },
        "default_verbosity": "moderate"
    }

    # Retry and timeout settings
    RETRY_CONFIG = {
        "max_retries": 5,  # Maximum retry attempts for API calls
        "base_delay": 15,  # Initial delay for retries (seconds)
        "max_delay": 120,  # Maximum delay between retries (seconds)
        "backoff_factor": 1.8,  # Exponential backoff multiplier
        "timeout": 90,  # Request timeout (seconds)
    }

    # Process pool settings
    PROCESS_POOL_CONFIG = {
        "max_workers": 4,  # Number of worker processes
        "use_spawn": True,  # Use spawn start method for compatibility
    }

    # Video search specific settings
    VIDEO_CONFIG = {
        "lightweight_processing": True,  # Skip ML models for video metadata
        "max_videos": 6,  # Maximum videos to process
        "youtube_domains": ["youtube.com", "youtu.be"],
    }

    # Logging configuration
    LOGGING_CONFIG = {
        "log_search_sessions": True,
        "log_format": "json",  # Use JSON for structured logging
        "log_compression_details": True,
        "log_cache_hits": True,
    }

    @classmethod
    def get_all_settings(cls):
        """Return all configuration settings as a dictionary"""
        return {
            "trusted_domains": cls.TRUSTED_DOMAINS,
            "models": {
                "embedding": cls.EMBEDDING_MODEL,
                "cross_encoder": cls.CROSS_ENCODER_MODEL,
                "query_expansion": cls.QUERY_EXPANSION_MODEL,
            },
            "search": cls.SEARCH_CONFIG,
            "rrf": cls.RRF_CONFIG,
            "compression": cls.COMPRESSION_CONFIG,
            "semantic_chunk": cls.SEMANTIC_CHUNK_CONFIG,
            "response": cls.RESPONSE_CONFIG,
            "retry": cls.RETRY_CONFIG,
            "process_pool": cls.PROCESS_POOL_CONFIG,
            "video": cls.VIDEO_CONFIG,
            "logging": cls.LOGGING_CONFIG,
        }
