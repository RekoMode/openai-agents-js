"""Ranks candidate alternates for each BOM part by cosine similarity."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def rank_alternates(
    embeddings: Dict[str, np.ndarray], top_k: int = 5
) -> Dict[str, List[Tuple[str, float]]]:
    """For every part, rank all other parts by cosine similarity.

    Args:
        embeddings: mapping of part_number -> (ideally L2-normalized) embedding.
        top_k: number of top alternates to keep per part.

    Returns:
        mapping of part_number -> ordered list of (alternate_part, similarity),
        highest similarity first, excluding the part itself.
    """
    part_numbers = list(embeddings.keys())
    matrix = np.stack([embeddings[p] for p in part_numbers])

    # Cosine similarity: normalize defensively in case embeddings weren't
    # already unit-length, then take the dot-product matrix.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    similarity = normalized @ normalized.T

    results: Dict[str, List[Tuple[str, float]]] = {}
    for i, part in enumerate(part_numbers):
        row = similarity[i].copy()
        row[i] = -np.inf  # exclude self
        order = np.argsort(-row)[:top_k]
        results[part] = [(part_numbers[j], float(row[j])) for j in order]
    return results
