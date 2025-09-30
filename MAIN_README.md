# SePA Web Search Pipeline

Implementation of the web search component from "SePA: A Search-enhanced Predictive Agent for Personalized Health Coaching" (IEEE BHI 2025).

## Repository Structure

**`paper-version/`** - Original implementation from the paper

- Single-query search, CPU-based processing
- Use for reproducing paper results

**`improved-version/`** - Enhanced production implementation

- Multi-query expansion, GPU acceleration, RRF ranking, improved fetching and processing of sources.
- Use for production deployments

## Key Enhancements

1. **Multi-Query Expansion**: Generates 4 diverse queries (general, scientific, demographic-specific, practical)
2. **Reciprocal Rank Fusion**: Merges results with k=60, 2x boost for trusted medical/sports domains
3. **GPU Acceleration**: 10x faster embeddings via HuggingFace Inference Endpoints
4. **Advanced Content Fetching**: Headless browser support for JavaScript-rendered content

## Usage

```python
# Paper version
from web_rag import create_rag
rag = create_rag()
response = await rag.process_query(query, user_context)

# Improved version
from src.web_search import RAG
from src.config import Config
rag = RAG(Config())
response = await rag.process_query(query, user_context)
```

## Citation

```bibtex
@inproceedings{sepa2025,
  title={SePA: A Search-enhanced Predictive Agent for Personalized Health Coaching},
  author={[Authors]},
  booktitle={IEEE International Conference on Biomedical and Health Informatics (BHI)},
  year={2025}
}
```

## Notes

- Paper version preserves the exact implementation for reproducibility
- Improved version includes production optimizations not described in the paper
- See individual directories for detailed documentation
