"""
Statistical association and weighting schemes:
- TF-IDF and sublinear term frequency scaling ($1 + \\log(\\text{tf})$)
- Positive Pointwise Mutual Information (PPMI) with $\\alpha$-context smoothing
- Shifted PPMI (SPPMI)
- Add-$k$ / Laplace co-occurrence smoothing
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer


def apply_sublinear_tf(matrix: csr_matrix) -> csr_matrix:
    """
    Apply sublinear term-frequency scaling to nonzero entries of a sparse count matrix:
    $$\\text{tf}_{\\text{sublinear}}(t, d) = 1 + \\log(\\text{tf}(t, d))$$ for $\\text{tf}(t, d) > 0$.
    """
    mat = matrix.tocsr(copy=True).astype(float)
    mat.data = 1.0 + np.log(mat.data)
    return mat


def apply_tfidf_weighting(
    matrix: csr_matrix,
    transformer: Optional[TfidfTransformer] = None,
    sublinear_tf: bool = False,
    fitting: bool = True,
) -> tuple[csr_matrix, TfidfTransformer]:
    """
    Fit or apply a `TfidfTransformer` with optional sublinear TF damping.
    """
    if transformer is None:
        transformer = TfidfTransformer(sublinear_tf=sublinear_tf)
    if fitting:
        transformer.fit(matrix)
    return transformer.transform(matrix), transformer


def laplace_smooth_cooccurrence(
    cooc_matrix: csr_matrix, k: float = 0.1
) -> np.ndarray:
    """
    Apply add-$k$ (Laplace) smoothing to a co-occurrence count matrix:
    $$C_{\\text{smooth}}(w, c) = C(w, c) + k$$
    """
    dense = cooc_matrix.toarray().astype(float) if hasattr(cooc_matrix, "toarray") else np.array(cooc_matrix, dtype=float)
    return dense + float(k)


def compute_ppmi_matrix(
    cooc_matrix: csr_matrix,
    alpha: float = 1.0,
    shift: float = 0.0,
    eps: float = 1e-12,
) -> csr_matrix:
    """
    Compute Positive Pointwise Mutual Information (PPMI) or Shifted PPMI (SPPMI)
    with $\\alpha$-smoothed context distribution $P_\\alpha(c)$ (Levy & Goldberg, 2014; Levy et al., 2015):

    $$P(w, c) = \\frac{C(w, c)}{\\sum_{w', c'} C(w', c')}, \\quad P(w) = \\frac{\\sum_{c'} C(w, c')}{\\sum_{w', c'} C(w', c')}$$
    $$P_\\alpha(c) = \\frac{\\left(\\sum_{w'} C(w', c)\\right)^\\alpha}{\\sum_{c'} \\left(\\sum_{w'} C(w', c')\\right)^\\alpha}$$
    $$\\text{PPMI}_\\alpha(w, c) = \\max\\left(0, \\log_2 \\frac{P(w, c)}{P(w) P_\\alpha(c)} - \\text{shift}\\right)$$
    """
    mat = cooc_matrix.tocsr().astype(float)
    total_count = float(mat.sum())
    if total_count <= 0:
        return csr_matrix(mat.shape, dtype=float)

    word_counts = np.asarray(mat.sum(axis=1)).flatten()
    context_counts = np.asarray(mat.sum(axis=0)).flatten()

    p_w = word_counts / total_count
    c_alpha = np.power(np.maximum(context_counts, 0.0), alpha)
    denom_c_alpha = float(c_alpha.sum())
    if denom_c_alpha <= 0:
        return csr_matrix(mat.shape, dtype=float)
    p_c_alpha = c_alpha / denom_c_alpha

    coo = mat.tocoo()
    p_wc = coo.data / total_count
    denom = np.maximum(p_w[coo.row] * p_c_alpha[coo.col], eps)
    pmi = np.log2(np.maximum(p_wc / denom, eps)) - shift
    ppmi_data = np.maximum(0.0, pmi)

    result = csr_matrix((ppmi_data, (coo.row, coo.col)), shape=mat.shape, dtype=float)
    result.eliminate_zeros()
    return result


def compute_sppmi_matrix(
    cooc_matrix: csr_matrix, k_negative: int = 5, alpha: float = 0.75
) -> csr_matrix:
    """
    Compute Shifted Positive Pointwise Mutual Information (SPPMI) shifted by $\\log_2(k)$:
    $$\\text{SPPMI}_k(w, c) = \\max\\left(0, \\text{PMI}_\\alpha(w, c) - \\log_2 k\\right)$$
    """
    shift = float(np.log2(max(k_negative, 1)))
    return compute_ppmi_matrix(cooc_matrix, alpha=alpha, shift=shift)
