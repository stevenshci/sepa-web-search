# SePA Web Search Pipeline

Implementation of the web search component from "SePA: A Search-enhanced Predictive Agent for Personalized Health Coaching" (IEEE BHI 2025).

## Repository Structure

**`paper-version/`** - Original implementation from the paper

- Single-query search, CPU-based processing
- Use for reproducing paper results

**`improved-search/`** - Improved implementation

- Multi-query expansion, GPU acceleration, RRF ranking, improved fetching and processing of sources.
- Use for production deployments

## Key Enhancements

1. **Multi-Query Expansion**: Generates 4 diverse queries (general, scientific, demographic-specific, practical)
2. **Reciprocal Rank Fusion**: Merges results with k=60, 2x boost for trusted medical/sports domains
3. **GPU Acceleration**: 10x faster embeddings via HuggingFace Inference Endpoints
4. **Advanced Content Fetching**: Headless browser support for JavaScript-rendered content

## Citation

```bibtex
@article{ozolcer2025sepa,
  title={SePA: A Search-enhanced Predictive Agent for Personalized Health Coaching},
  author={Ozolcer, Melik and Bae, Sang Won},
  journal={arXiv preprint arXiv:2509.04752},
  year={2025}
}
```

## Notes

- Paper version preserves the exact implementation for reproducibility
- Improved version includes production optimizations not described in the paper
- See individual directories for detailed documentation


