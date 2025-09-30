from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

class RecipRocalRankFusion:
    """
    Implementation of Reciprocal Rank Fusion with domain expertise boosting.
    """

    def __init__(self, k: int = 60, boost_factor: float = 2.0):
        """
        Initialize RRF with parameters.
        """
        self.k = k
        self.boost_factor = boost_factor

    def fuse(
        self,
        result_lists: List[List[Dict[str, Any]]],
        trusted_domains: Optional[List[str]] = None,
        return_scores: bool = False
    ) -> List[Dict[str, Any]] | List[Tuple[Dict[str, Any], float]]:
        """
        Fuse multiple ranked result lists using RRF with domain boosting.

        Args:
            result_lists: List of ranked result lists, each containing dicts with 'link' field
            trusted_domains: List of trusted domain strings to boost
            return_scores: If True, return tuples of (result, score)

        Returns:
            Merged and re-ranked list of results (optionally with scores)
        """
        if not result_lists:
            return []

        # Dictionary to store RRF scores
        rrf_scores = {}

        # Calculate RRF score for each document across all lists
        for list_idx, result_list in enumerate(result_lists):
            for rank, result in enumerate(result_list):
                # Use URL as unique identifier
                doc_id = result.get('link', '')
                if not doc_id:
                    continue

                # Base RRF formula: 1 / (k + rank)
                # rank+1 because enumerate starts at 0
                base_score = 1 / (self.k + rank + 1)

                # Apply domain boosting if applicable
                score = base_score
                if trusted_domains and self._is_trusted_domain(doc_id, trusted_domains):
                    score *= self.boost_factor

                # Aggregate scores across lists
                if doc_id in rrf_scores:
                    rrf_scores[doc_id]['score'] += score
                    rrf_scores[doc_id]['occurrences'] += 1
                    # Keep the result with most complete metadata
                    if len(result) > len(rrf_scores[doc_id]['result']):
                        rrf_scores[doc_id]['result'] = result
                else:
                    rrf_scores[doc_id] = {
                        'score': score,
                        'result': result,
                        'is_trusted': self._is_trusted_domain(doc_id, trusted_domains) if trusted_domains else False,
                        'occurrences': 1
                    }

        # Sort by RRF score in descending order
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )

        # Log statistics
        if sorted_results:
            self._log_statistics(sorted_results, len(result_lists))

        # Return results with or without scores
        if return_scores:
            return [(item[1]['result'], item[1]['score']) for item in sorted_results]
        else:
            return [item[1]['result'] for item in sorted_results]

    def _is_trusted_domain(self, url: str, trusted_domains: List[str]) -> bool:
        """Check if URL belongs to a trusted domain."""
        if not trusted_domains:
            return False
        return any(domain in url for domain in trusted_domains)

    def _log_statistics(self, sorted_results: List, num_lists: int):
        """Log statistics about the fusion results."""
        total_results = len(sorted_results)
        if total_results == 0:
            return

        top_score = sorted_results[0][1]['score']

        # Count trusted domains in top 10
        top_10 = sorted_results[:10] if len(sorted_results) >= 10 else sorted_results
        trusted_count = sum(1 for item in top_10 if item[1]['is_trusted'])

        # Calculate occurrence statistics
        avg_occurrences = sum(item[1]['occurrences'] for item in sorted_results) / total_results
        max_occurrences = max(item[1]['occurrences'] for item in sorted_results)

        # Log top 5 results
        for i, (url, data) in enumerate(sorted_results[:5]):
            title = data['result'].get('title', 'No title')[:50]

def calculate_rrf_weights(num_lists: int, k: int = 60) -> Dict[int, float]:
    """
    Calculate the theoretical weight contribution of each rank position.

    This helps understand how much each position contributes to final scores.

    Args:
        num_lists: Number of result lists being fused
        k: RRF constant

    Returns:
        Dictionary mapping rank to weight percentage
    """
    weights = {}
    total_weight = 0

    # Calculate weights for first 20 positions
    for rank in range(20):
        weight = 1 / (k + rank + 1)
        weights[rank + 1] = weight
        total_weight += weight * num_lists  # Assuming uniform distribution

    # Convert to percentages
    weight_percentages = {
        rank: (weight / total_weight) * 100
        for rank, weight in weights.items()
    }

    return weight_percentages

def analyze_domain_distribution(
    results: List[Dict[str, Any]],
    trusted_domains: List[str]
) -> Dict[str, Any]:
    """
    Analyze the distribution of trusted vs non-trusted domains in results.

    Args:
        results: List of search results
        trusted_domains: List of trusted domain strings

    Returns:
        Statistics about domain distribution
    """
    trusted_count = 0
    domain_counts = defaultdict(int)

    for result in results:
        url = result.get('link', '')
        is_trusted = any(domain in url for domain in trusted_domains)

        if is_trusted:
            trusted_count += 1
            # Find which trusted domain
            for domain in trusted_domains:
                if domain in url:
                    domain_counts[domain] += 1
                    break

    return {
        'total_results': len(results),
        'trusted_count': trusted_count,
        'trusted_percentage': (trusted_count / len(results) * 100) if results else 0,
        'top_domains': dict(sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
        'unique_trusted_domains': len(domain_counts)
    }
