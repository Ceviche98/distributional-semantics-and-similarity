"""
Vector Space Semantics for Similarity & Character Profiling.

A modular distributional semantics package providing:
- Corpus aggregation, contraction expansion, negation-aware stopword filtering, and lexical normalization (`corpus`).
- Sparse document-term and sliding-window word-context co-occurrence matrix construction (`cooccurrence`).
- Statistical association & weighting schemes including TF-IDF, sublinear TF scaling, PPMI, Shifted PPMI, and alpha-smoothed context distributions (`weighting`).
- Dimensionality reduction via Truncated SVD with eigenvalue exponentiation, L2 normalization, and cosine similarity (`factorization`).
- Dense syntactic (POS) and neural (`SentenceTransformer`) embeddings with hybrid sparse-dense multi-matrix fusion (`embeddings`).
- Information retrieval evaluation, grid search optimization, and feature-level linguistic attribution (`evaluation`).
"""

from .corpus import (
    CONTRACTION_MAP,
    NEGATION_WORDS_SET,
    VADER_NEGATE,
    build_vocabulary,
    create_character_document_from_dataframe,
    expand_contractions,
    load_script_dataset,
    pre_process,
    pre_process_baseline,
    pre_process_raw,
)
from .cooccurrence import (
    build_cooccurrence_matrix,
    build_word_and_char_matrices,
    generate_ngrams,
    get_all_ngrams,
    to_feature_vector_dictionary,
)
from .weighting import (
    apply_sublinear_tf,
    apply_tfidf_weighting,
    compute_ppmi_matrix,
    compute_sppmi_matrix,
    laplace_smooth_cooccurrence,
)
from .factorization import (
    apply_truncated_svd,
    compute_cosine_similarity,
    compute_similarity_matrix,
    l2_normalize_matrix,
)
from .embeddings import (
    combine_weighted_matrices,
    extract_pos_matrix,
    get_avg_document_embedding,
    get_pos_features,
    hybrid_score_ensemble,
)
from .evaluation import (
    analyze_feature_contribution,
    analyze_missing_feature_contribution,
    compute_IR_evaluation_scores,
    create_document_matrix_from_corpus,
    plot_heat_map_similarity,
    run_config_features,
    run_preprocessing_grid_search,
)

__all__ = [
    "CONTRACTION_MAP",
    "NEGATION_WORDS_SET",
    "VADER_NEGATE",
    "build_vocabulary",
    "create_character_document_from_dataframe",
    "expand_contractions",
    "load_script_dataset",
    "pre_process",
    "pre_process_baseline",
    "pre_process_raw",
    "build_cooccurrence_matrix",
    "build_word_and_char_matrices",
    "generate_ngrams",
    "get_all_ngrams",
    "to_feature_vector_dictionary",
    "apply_sublinear_tf",
    "apply_tfidf_weighting",
    "compute_ppmi_matrix",
    "compute_sppmi_matrix",
    "laplace_smooth_cooccurrence",
    "apply_truncated_svd",
    "compute_cosine_similarity",
    "compute_similarity_matrix",
    "l2_normalize_matrix",
    "combine_weighted_matrices",
    "extract_pos_matrix",
    "get_avg_document_embedding",
    "get_pos_features",
    "hybrid_score_ensemble",
    "analyze_feature_contribution",
    "analyze_missing_feature_contribution",
    "compute_IR_evaluation_scores",
    "create_document_matrix_from_corpus",
    "plot_heat_map_similarity",
    "run_config_features",
    "run_preprocessing_grid_search",
]
