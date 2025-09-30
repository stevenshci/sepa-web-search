## Overview

Enhanced implementation of the SePA web search component with production optimizations. This version includes multi-query expansion, GPU acceleration, Reciprocal Rank Fusion, and advanced content fetching capabilities not present in the original paper implementation.

The system implements a Web-Search Retrieval-Augmented Generation (RAG) pipeline optimized for sports science and health/exercise information retrieval, with domain-specific enhancements for athletic performance queries.

## System Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  User Query  │────▶│Query Expansion│────▶│ 4 Parallel   │
└──────────────┘     │  (LLM-based) │     │   Searches   │
                     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
                                          ┌──────────────┐
                                          │   RRF with   │
                                          │Domain Boost  │
                                          └──────┬───────┘
                                                  │
┌──────────────┐     ┌──────────────┐            ▼
│   Response   │◀────│   Context    │◀────┌──────────────┐
│  Generation  │     │ Compression  │     │   Document   │
└──────────────┘     └──────────────┘     │  Processing  │
                                          └──────────────┘
```

### Pipeline Stages

1. **Query Expansion**: Transform single query into 4 diverse search queries
2. **Parallel Search**: Execute searches simultaneously via Google Custom Search API
3. **Result Fusion**: Merge results using RRF with trusted domain boosting
4. **Content Processing**: Fetch, extract, and chunk documents semantically
5. **Reranking**: Cross-encoder scoring for relevance assessment
6. **Compression**: Reduce context while preserving key information
7. **Generation**: Produce response with inline citations

### Prerequisites

- Python 3.11
- Google Custom Search Engine API access
- OpenAI Compatible LLM API key
- HuggingFace account with Inference Endpoints deployed

## Repository Structure

```
improved-version/
├── src/                    # Core implementation
│   ├── config.py          # Configuration and trusted domains
│   ├── web_search.py      # Main search pipeline
│   ├── response.py        # LLM response generation
│   ├── inference.py       # HuggingFace inference endpoints
│   ├── embeddings.py      # Embedding interface
│   └── ranking.py         # RRF ranking/fusion
├── examples/
│   └── demo.ipynb        # Complete pipeline demo
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
└── README.md            # This file
```

## Demo

See [examples/demo.ipynb](examples/demo.ipynb) for an interactive demonstration with full pipeline visualization.

## Key Features

### Models & APIs

- **Query Expansion**: LLM API call for diverse query generation
- **Response Generation**: Compatible with any OpenAI API compatible LLM
- **Embeddings**: BAAI/bge-base-en-v1.5 via HuggingFace
- **Reranking**: BAAI/bge-reranker-base for relevance scoring

### Content Processing

- **Web Scraping**: Playwright for JavaScript-heavy sites
- **Text Extraction**: Three-stage pipeline (Readability → Trafilatura → BeautifulSoup)
- **PDF Support**: PyPDF2 for academic paper extraction
- **Semantic Chunking**: Sentence-based splitting with 1000 char chunks

### Special Handling

- **YouTube Videos**: Included as references with metadata
- **Reddit Threads**: Enhanced JavaScript rendering with extended wait
- **Academic Papers**: PDF extraction from PubMed/journals
- **Fallback Content**: Uses search snippets when fetch fails

> **Note**: Current configuration uses 4 queries, processes 32 documents/batch, and achieves ~40% content fetch success. These parameters can be increased with additional API quotas and computational resources for improved coverage and performance.
