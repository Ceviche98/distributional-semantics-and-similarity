"""
Comprehensive unit and integration test suite for `vector_semantics`.
Tests:
1. Corpus preprocessing, contraction expansion, negation-preserving stopword filtering, and vocabulary scaling.
2. Sliding-window sparse co-occurrence matrix construction (symmetric, directional, inverse-distance weighted) and n-gram dictionaries.
3. Statistical association weighting: sublinear TF scaling, TF-IDF, Laplace smoothing, Positive PMI (PPMI), alpha-context smoothing, and Shifted PPMI (SPPMI).
4. Truncated SVD dimensionality reduction with eigenvalue exponentiation ($U_k \\Sigma_k^\\gamma$), L2 normalization, and cosine similarity ranking.
5. Hybrid sparse-dense matrix fusion, POS syntactic feature extraction, and IR evaluation accuracy on synthetic and real EastEnders script data.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from vector_semantics import (
    analyze_feature_contribution,
    analyze_missing_feature_contribution,
    apply_sublinear_tf,
    apply_tfidf_weighting,
    apply_truncated_svd,
    build_cooccurrence_matrix,
    build_vocabulary,
    combine_weighted_matrices,
    compute_IR_evaluation_scores,
    compute_cosine_similarity,
    compute_ppmi_matrix,
    compute_similarity_matrix,
    compute_sppmi_matrix,
    create_character_document_from_dataframe,
    create_document_matrix_from_corpus,
    expand_contractions,
    extract_pos_matrix,
    generate_ngrams,
    get_all_ngrams,
    get_pos_features,
    hybrid_score_ensemble,
    l2_normalize_matrix,
    laplace_smooth_cooccurrence,
    load_script_dataset,
    pre_process,
    pre_process_baseline,
    pre_process_raw,
    run_preprocessing_grid_search,
    to_feature_vector_dictionary,
)


def test_contraction_expansion_and_preprocessing():
    text = "I don't think you're going... wait 123!"
    expanded = expand_contractions(text)
    assert "do not" in expanded.lower()
    assert "you are" in expanded.lower()

    # Keep ellipsis and negation words, remove digits and standard punctuation except ! and ?
    cfg_keep_ellip = {
        "lower_case": True,
        "erase_numbers": True,
        "erase_ellipsis": False,
        "contractions": True,
        "erase_stop_words": True,
        "erase_stop_words_negation": True,
        "punctuation": True,
        "split": "tokenizer",
        "lex_norm": "lemmatization",
    }
    tokens = pre_process(text, cfg_keep_ellip)
    assert "not" in tokens
    assert "..." in tokens
    assert "!" in tokens
    assert "123" not in tokens

    # Erase ellipsis when configured
    cfg_no_ellip = {**cfg_keep_ellip, "erase_ellipsis": True}
    tokens_no_ellip = pre_process(text, cfg_no_ellip)
    assert "..." not in tokens_no_ellip


def test_vocabulary_and_ngrams():
    docs = [
        ["eastenders", "pub", "queen", "vic", "pub"],
        ["queen", "vic", "london", "pub"],
    ]
    vocab, counts = build_vocabulary(docs, min_freq=2)
    assert "pub" in vocab
    assert "queen" in vocab
    assert "vic" in vocab
    assert "london" not in vocab
    assert counts["pub"] == 3

    bigrams = generate_ngrams(["not", "good", "mate"], 2)
    assert bigrams == ["not_good", "good_mate"]

    all_ng = get_all_ngrams(["a", "b", "c"], max_n=2, include_unigrams=True)
    assert all_ng == ["a", "b", "c", "a_b", "b_c"]

    feat_dict = to_feature_vector_dictionary(["a", "b", "a"], ngram=1, extra_features=["extra_tok"])
    assert feat_dict["a"] == 2
    assert feat_dict["b"] == 1
    assert feat_dict["extra_tok"] == 1


def test_sliding_window_cooccurrence_matrix():
    docs = [
        ["tea", "coffee", "cup", "tea", "cup"],
        ["coffee", "mug", "cup"],
    ]
    cooc_sym, vocab = build_cooccurrence_matrix(docs, window_size=1, symmetric=True)
    assert cooc_sym.shape == (len(vocab), len(vocab))
    # Symmetric matrix check
    diff = np.abs((cooc_sym - cooc_sym.T).toarray()).max()
    assert diff == pytest.approx(0.0)

    # Tea and coffee co-occur adjacent once in doc 0
    t_idx, c_idx = vocab["tea"], vocab["coffee"]
    assert cooc_sym[t_idx, c_idx] == pytest.approx(1.0)
    assert cooc_sym[c_idx, t_idx] == pytest.approx(1.0)

    # Directional co-occurrence should only count right-hand contexts
    cooc_dir, _ = build_cooccurrence_matrix(docs, window_size=1, vocab=vocab, symmetric=False)
    assert cooc_dir[t_idx, c_idx] == pytest.approx(1.0)
    assert cooc_dir[c_idx, t_idx] == pytest.approx(0.0)


def test_ppmi_sppmi_and_weighting():
    # Create a block-diagonal co-occurrence matrix where pairs (0,1) and (2,3) strongly co-occur
    counts = csr_matrix(
        np.array(
            [
                [10.0, 9.0, 0.0, 0.0],
                [9.0, 10.0, 0.0, 0.0],
                [0.0, 0.0, 8.0, 8.0],
                [0.0, 0.0, 8.0, 8.0],
            ]
        )
    )
    ppmi = compute_ppmi_matrix(counts, alpha=1.0)
    dense_ppmi = ppmi.toarray()
    assert np.all(dense_ppmi >= 0.0)
    assert dense_ppmi[0, 1] > 0.5
    assert dense_ppmi[0, 2] == pytest.approx(0.0)

    # Alpha-smoothed PPMI and Shifted PPMI (SPPMI)
    ppmi_alpha = compute_ppmi_matrix(counts, alpha=0.75)
    sppmi = compute_sppmi_matrix(counts, k_negative=2, alpha=0.75)
    assert sppmi[0, 1] < ppmi_alpha[0, 1]

    # Sublinear TF scaling: 1 + log(e) == 2.0
    sub_tf = apply_sublinear_tf(csr_matrix([[1.0, np.e]]))
    assert sub_tf.toarray()[0, 0] == pytest.approx(1.0)
    assert sub_tf.toarray()[0, 1] == pytest.approx(2.0)

    # Laplace smoothing
    smoothed = laplace_smooth_cooccurrence(counts, k=0.5)
    assert smoothed[0, 2] == pytest.approx(0.5)
    assert smoothed[0, 0] == pytest.approx(10.5)

    # TF-IDF weighting
    tfidf_mat, _ = apply_tfidf_weighting(counts, sublinear_tf=True, fitting=True)
    assert tfidf_mat.shape == counts.shape


def test_truncated_svd_and_cosine_similarity():
    rng = np.random.default_rng(42)
    mat = csr_matrix(rng.uniform(0, 5, size=(12, 25)))

    reduced_g1, svd_model = apply_truncated_svd(mat, n_components=5, gamma=1.0, normalize_output=True)
    assert reduced_g1.shape == (12, 5)
    assert svd_model is not None
    norms = np.linalg.norm(reduced_g1, axis=1)
    assert np.allclose(norms, 1.0)

    # Eigenvalue exponentiation gamma = 0.5
    reduced_g05, _ = apply_truncated_svd(mat, n_components=5, gamma=0.5, normalize_output=True)
    assert reduced_g05.shape == (12, 5)
    assert np.allclose(np.linalg.norm(reduced_g05, axis=1), 1.0)

    # Cosine similarity properties
    v1 = np.array([1.0, 2.0, 3.0])
    assert compute_cosine_similarity(v1, v1) == pytest.approx(1.0)
    assert compute_cosine_similarity(v1, -v1) == pytest.approx(-1.0)
    assert compute_cosine_similarity(np.zeros(3), v1) == pytest.approx(0.0)

    sim_mat = compute_similarity_matrix(reduced_g1, reduced_g1)
    assert sim_mat.shape == (12, 12)
    assert np.allclose(np.diag(sim_mat), 1.0)


def test_pos_features_and_hybrid_ensembling():
    sample_doc = "Where have you been? _EOL_ I told you I was busy! _EOL_ Look at this good club. _EOL_"
    tokens = pre_process_raw(sample_doc)
    feats = get_pos_features(sample_doc, tokens)
    assert len(feats) == 10
    assert feats[0] > 0  # avg_word_per_sentence
    assert feats[1] > 0  # question mark ratio

    pos_mat = extract_pos_matrix([sample_doc, sample_doc], [tokens, tokens])
    assert pos_mat.shape == (2, 10)

    m1 = csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    m2 = csr_matrix([[0.0, 2.0], [3.0, 0.0]])
    combined = combine_weighted_matrices([m1, m2], [0.8, 0.7], l2_normalize_each=True)
    assert combined.shape == (2, 4)

    ens = hybrid_score_ensemble(np.eye(2), np.ones((2, 2)), alpha=0.8)
    assert ens[0, 0] == pytest.approx(1.0)
    assert ens[0, 1] == pytest.approx(0.2)


def test_synthetic_ir_evaluation_and_grid_search():
    df_train = pd.DataFrame(
        {
            "Character_name": ["ALICE", "ALICE", "BOB", "BOB", "CHARLIE", "CHARLIE"],
            "Gender": ["F", "F", "M", "M", "M", "M"],
            "Line": [
                "Hospital doctor medicine surgery patient.",
                "Surgery ward clinical nurse doctor!",
                "Football stadium goal striker match?",
                "Match referee penalty goal striker.",
                "Kitchen recipe chef oven baking...",
                "Baking pastry chef dinner kitchen...",
            ],
        }
    )
    df_val = pd.DataFrame(
        {
            "Character_name": ["ALICE", "BOB", "CHARLIE"],
            "Gender": ["F", "M", "M"],
            "Line": [
                "Clinical doctor surgery patient ward!",
                "Striker football goal match penalty?",
                "Chef kitchen baking recipe oven...",
            ],
        }
    )

    train_docs = create_character_document_from_dataframe(df_train, max_line_count=10)
    val_docs = create_character_document_from_dataframe(df_val, max_line_count=10)

    vectorizers = {}
    cfg_feat = {
        "ngrams": 1,
        "td-idf": True,
        "char_min_df": 1,
        "charmin_range": 3,
        "charmax_range": 4,
        "w_uni": 0.8,
        "w_char": 0.7,
        "sublinear_tf": True,
    }
    train_mat = create_document_matrix_from_corpus(train_docs, vectorizers, config_features=cfg_feat, fitting=True)
    val_mat = create_document_matrix_from_corpus(val_docs, vectorizers, config_features=cfg_feat, fitting=False)

    labels = sorted(train_docs.keys())
    mean_rank, mean_sim, acc, sim_df = compute_IR_evaluation_scores(train_mat, val_mat, labels, labels)
    assert mean_rank == pytest.approx(1.0)
    assert acc == pytest.approx(1.0)
    assert mean_sim > 0.3
    assert len(sim_df) == 9

    contrib_df = analyze_feature_contribution("ALICE", "ALICE", val_mat, train_mat, labels, labels, vectorizers)
    assert not contrib_df.empty
    missing_df = analyze_missing_feature_contribution("ALICE", "ALICE", val_mat, train_mat, labels, labels, vectorizers)
    assert not missing_df.empty

    grid_res = run_preprocessing_grid_search(
        train_docs,
        val_docs,
        {
            "lower_case": [True],
            "erase_numbers": [True],
            "erase_ellipsis": [False],
            "contractions": [True],
            "erase_stop_words": [True],
            "erase_stop_words_negation": [True],
            "punctuation": [True],
            "split": ["split"],
            "lex_norm": ["none", "stemming"],
        },
    )
    assert len(grid_res) == 2


def test_real_dataset_reproducibility():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    if not os.path.exists(os.path.join(data_dir, "training.csv")):
        pytest.skip("Real EastEnders dataset not found in data/")

    train_df = load_script_dataset(os.path.join(data_dir, "training.csv"))
    val_df = load_script_dataset(os.path.join(data_dir, "val.csv"))
    test_df = load_script_dataset(os.path.join(data_dir, "test.csv"))

    train_docs = create_character_document_from_dataframe(train_df, max_line_count=300)
    val_docs = create_character_document_from_dataframe(val_df, max_line_count=50)
    test_docs = create_character_document_from_dataframe(test_df, max_line_count=50)

    train_labels = sorted(train_docs.keys())
    val_labels = sorted(val_docs.keys())
    test_labels = sorted(test_docs.keys())

    vectorizers = {}
    train_mat = create_document_matrix_from_corpus(train_docs, vectorizers, fitting=True)
    val_mat = create_document_matrix_from_corpus(val_docs, vectorizers, fitting=False)
    test_mat = create_document_matrix_from_corpus(test_docs, vectorizers, fitting=False)

    val_rank, _, val_acc, _ = compute_IR_evaluation_scores(train_mat, val_mat, train_labels, val_labels)
    test_rank, _, test_acc, _ = compute_IR_evaluation_scores(train_mat, test_mat, train_labels, test_labels)

    # Exact reproduction of CW2___NLP.pdf paper metrics:
    # Validation: Mean Rank 1.125, Accuracy 0.9375 (15/16)
    # Blind Test: Mean Rank 1.125, Accuracy 0.875 (14/16)
    assert val_rank == pytest.approx(1.125)
    assert val_acc == pytest.approx(0.9375)
    assert test_rank == pytest.approx(1.125)
    assert test_acc == pytest.approx(0.875)
