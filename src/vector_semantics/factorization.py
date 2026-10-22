"""
Matrix factorization via Truncated Singular Value Decomposition (SVD),
eigenvalue exponentiation ($U_k \\Sigma_k^\\gamma$), L2 vector normalization, and cosine similarity.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy.linalg import norm
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def l2_normalize_matrix(matrix: Union[csr_matrix, np.ndarray]) -> Union[csr_matrix, np.ndarray]:
    """Normalize rows of a sparse or dense matrix to unit L2 norm."""
    return normalize(matrix, norm="l2", axis=1)


def apply_truncated_svd(
    matrix: Union[csr_matrix, np.ndarray],
    n_components: int = 300,
    gamma: float = 1.0,
    normalize_output: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, Optional[TruncatedSVD]]:
    """
    Perform Truncated Singular Value Decomposition (SVD) on a sparse representation matrix $M$:
    $$M \\approx U_k \\Sigma_k V_k^T, \\quad W_{\\text{SVD}} = U_k \\Sigma_k^\\gamma$$

    When $\\gamma = 1.0$, this matches standard `TruncatedSVD.fit_transform(M) = U_k \\Sigma_k$.
    When $\\gamma \\in [0, 1)$ (e.g., $\\gamma = 0.5$ or $\\gamma = 0.0$), singular values are dampened
    as studied in distributional semantics (Caron, 2001; Levy et al., 2015).
    """
    max_k = min(matrix.shape[0] - 1, matrix.shape[1] - 1)
    k = max(1, min(n_components, max_k)) if max_k >= 1 else 1

    if gamma == 1.0 and min(matrix.shape) > k:
        svd = TruncatedSVD(n_components=k, random_state=random_state)
        reduced = svd.fit_transform(matrix)
        if normalize_output:
            reduced = normalize(reduced, norm="l2")
        return reduced, svd

    mat_float = matrix.astype(float)
    if issparse(mat_float) and k < min(mat_float.shape) - 1:
        u, s, _ = svds(mat_float, k=k, random_state=random_state)
        order = np.argsort(-s)
        u = u[:, order]
        s = s[order]
    else:
        dense = mat_float.toarray() if issparse(mat_float) else np.asarray(mat_float, dtype=float)
        u_full, s_full, _ = np.linalg.svd(dense, full_matrices=False)
        u = u_full[:, :k]
        s = s_full[:k]

    s_weighted = np.power(np.maximum(s, 0.0), gamma)
    reduced = u * s_weighted
    if normalize_output:
        reduced = normalize(reduced, norm="l2")
    return reduced, None


def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1D vectors $v_1$ and $v_2$:
    $$\\text{sim}_{\\cos}(v_1, v_2) = \\frac{v_1 \\cdot v_2}{\\|v_1\\|_2 \\|v_2\\|_2}$$
    """
    arr1 = np.asarray(v1, dtype=float).flatten()
    arr2 = np.asarray(v2, dtype=float).flatten()
    denom = float(norm(arr1) * norm(arr2))
    if denom == 0.0:
        return 0.0
    return float(np.dot(arr1, arr2) / denom)


def compute_similarity_matrix(
    query_matrix: Union[csr_matrix, np.ndarray],
    target_matrix: Union[csr_matrix, np.ndarray],
) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix $S \\in \\mathbb{R}^{N_{\\text{query}} \\times N_{\\text{target}}}$.
    """
    q_norm = normalize(query_matrix, norm="l2")
    t_norm = normalize(target_matrix, norm="l2")
    sim = q_norm @ t_norm.T
    return sim.toarray() if issparse(sim) else np.asarray(sim, dtype=float)
