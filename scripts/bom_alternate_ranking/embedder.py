"""Embeds part spec/datasheet text with sentence-transformers."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def embed_parts(texts: Dict[str, str], model_name: str = "all-MiniLM-L6-v2") -> Dict[str, np.ndarray]:
    """Embed each part's text.

    Args:
        texts: mapping of part_number -> spec/datasheet text.
        model_name: sentence-transformers model id.

    Returns:
        mapping of part_number -> L2-normalized embedding vector.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    part_numbers: List[str] = list(texts.keys())
    corpus = [texts[p] or p for p in part_numbers]  # never embed an empty string

    embeddings = model.encode(
        corpus,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return {part: vec for part, vec in zip(part_numbers, embeddings)}
