"""
Information Retrieval (IR) evaluation harness, optimal multi-matrix pipeline,
grid-search orchestration, and element-wise feature attribution analysis.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .cooccurrence import to_feature_vector_dictionary
from .corpus import DEFAULT_PREPROCESS_CONFIG, pre_process
from .factorization import compute_cosine_similarity

OPTIMAL_PREPROCESS_CONFIG: Dict[str, Any] = {
    "lower_case": True,
    "erase_numbers": True,
    "erase_ellipsis": False,
    "contractions": True,
    "erase_stop_words": True,
    "erase_stop_words_negation": True,
    "punctuation": True,
    "split": "tokenizer",
    "lex_norm": None,
}

OPTIMAL_FEATURES_CONFIG: Dict[str, Any] = {
    "ngrams": 1,
    "td-idf": True,
    "char_min_df": 3,
    "charmin_range": 4,
    "charmax_range": 4,
    "w_uni": 0.8,
    "w_char": 0.7,
    "sublinear_tf": True,
}


def compute_IR_evaluation_scores(
    train_feature_matrix: Union[csr_matrix, np.ndarray],
    test_feature_matrix: Union[csr_matrix, np.ndarray],
    train_labels: List[str],
    test_labels: List[str],
    verbose: bool = False,
) -> Tuple[float, float, float, pd.DataFrame]:
    """
    Compute Information Retrieval (IR) character classification metrics:
    - `mean_rank`: Mean rank of the true character document (1.0 is optimal, 16.0 is worst).
    - `mean_cosine_similarity`: Mean cosine similarity between matched validation/test and training character vectors.
    - `accuracy`: Proportion of held-out character documents ranked at Top-1 (`rank == 1`).
    - `df`: DataFrame of all pairwise `(doc1, doc2, similarity)` scores.
    """
    rankings: List[int] = []
    all_cosine_similarities: List[float] = []
    pairwise_cosine_similarity: List[float] = []
    pairs: List[Tuple[str, str]] = []
    correct = 0

    train_dense = (
        train_feature_matrix.toarray()
        if issparse(train_feature_matrix)
        else np.asarray(train_feature_matrix, dtype=float)
    )
    test_dense = (
        test_feature_matrix.toarray()
        if issparse(test_feature_matrix)
        else np.asarray(test_feature_matrix, dtype=float)
    )

    for i, target in enumerate(test_labels):
        fm_1 = test_dense[i]
        all_sims: Dict[str, float] = {}
        for j, other in enumerate(train_labels):
            fm_2 = train_dense[j]
            sim = compute_cosine_similarity(fm_1, fm_2)
            pairs.append((target, other))
            pairwise_cosine_similarity.append(sim)
            if other == target:
                all_cosine_similarities.append(sim)
            all_sims[other] = sim

        sorted_similarities = sorted(all_sims.items(), key=lambda x: x[1], reverse=True)
        ranking = {key[0]: rank for rank, key in enumerate(sorted_similarities, 1)}
        if ranking[target] == 1:
            correct += 1
        rankings.append(ranking[target])

    mean_rank = float(np.mean(rankings))
    mean_cosine_similarity = float(np.mean(all_cosine_similarities))
    accuracy = float(correct / len(test_labels))

    if verbose:
        print("mean rank", mean_rank)
        print("mean cosine similarity", mean_cosine_similarity)
        print(correct, "correct out of", len(test_labels), "/ accuracy:", accuracy)

    df = pd.DataFrame(
        {
            "doc1": [x[0] for x in pairs],
            "doc2": [x[1] for x in pairs],
            "similarity": pairwise_cosine_similarity,
        }
    )
    return mean_rank, mean_cosine_similarity, accuracy, df


def generate_config_ngrams(
    corpus_preprocessed: List[List[str]],
    corpus_vectorizer: DictVectorizer,
    tfidf_transformer: TfidfTransformer,
    config: Dict[str, Any],
    fitting: bool = False,
) -> csr_matrix:
    """Vectorize preprocessed token lists into a count or TF-IDF sparse matrix."""
    dicts_ngrams = [
        to_feature_vector_dictionary(doc, config.get("ngrams", 1))
        for doc in corpus_preprocessed
    ]
    if fitting:
        corpus_vectorizer.fit(dicts_ngrams)
    matrix = corpus_vectorizer.transform(dicts_ngrams)

    if config.get("td-idf", True):
        if fitting:
            tfidf_transformer.fit(matrix)
        matrix = tfidf_transformer.transform(matrix)
    return matrix


def create_document_matrix_from_corpus(
    corpus_character_docs: Dict[str, str],
    vectorizers: Dict[str, Any],
    config_preprocess: Optional[Dict[str, Any]] = None,
    config_features: Optional[Dict[str, Any]] = None,
    fitting: bool = False,
) -> csr_matrix:
    """
    Construct the optimal multi-view sparse document representation (Question 4 & Question 5):
    - Channel 1: Preprocessed word unigrams weighted by TF-IDF and L2-normalized, scaled by $w_{\\text{uni}} = 0.8$.
    - Channel 2: Raw lowercase character quadrigrams ($n = 4, \\text{min\\_df} = 3$) with sublinear TF scaling
      ($1 + \\log(\\text{tf})$), IDF weighting, and L2 normalization, scaled by $w_{\\text{char}} = 0.7$.
    """
    cfg_prep = {**OPTIMAL_PREPROCESS_CONFIG, **(config_preprocess or {})}
    cfg_feat = {**OPTIMAL_FEATURES_CONFIG, **(config_features or {})}

    if fitting:
        vectorizers["word_dict"] = DictVectorizer()
        vectorizers["word_tfidf"] = TfidfTransformer(
            sublinear_tf=cfg_feat.get("sublinear_tf_word", False)
        )

    corpus_preprocessed = [
        pre_process(doc, cfg_prep) for _, doc in sorted(corpus_character_docs.items())
    ]

    matrix_uni = generate_config_ngrams(
        corpus_preprocessed,
        vectorizers["word_dict"],
        vectorizers["word_tfidf"],
        cfg_feat,
        fitting=fitting,
    )
    matrix_uni = normalize(matrix_uni, norm="l2")

    corpus_raw = [doc.lower() for _, doc in sorted(corpus_character_docs.items())]
    if fitting:
        vectorizers["char_tdidf"] = TfidfVectorizer(
            analyzer="char",
            ngram_range=(cfg_feat["charmin_range"], cfg_feat["charmax_range"]),
            min_df=cfg_feat["char_min_df"],
            sublinear_tf=cfg_feat["sublinear_tf"],
        )
        vectorizers["char_tdidf"].fit(corpus_raw)

    matrix_char = vectorizers["char_tdidf"].transform(corpus_raw)
    matrix_char = normalize(matrix_char, norm="l2")

    final_matrix = hstack(
        [matrix_uni * cfg_feat["w_uni"], matrix_char * cfg_feat["w_char"]]
    ).tocsr()
    return final_matrix


def run_config_features(
    training_character_docs: Dict[str, str],
    validation_character_docs: Dict[str, str],
    config_preprocess: Optional[Dict[str, Any]] = None,
    config_features: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, float, pd.DataFrame]:
    """Run a single preprocessing + feature extraction configuration and return validation IR scores."""
    vectorizers: Dict[str, Any] = {}
    train_labels = sorted(training_character_docs.keys())
    val_labels = sorted(validation_character_docs.keys())

    train_mat = create_document_matrix_from_corpus(
        training_character_docs, vectorizers, config_preprocess, config_features, fitting=True
    )
    val_mat = create_document_matrix_from_corpus(
        validation_character_docs, vectorizers, config_preprocess, config_features, fitting=False
    )
    return compute_IR_evaluation_scores(train_mat, val_mat, train_labels, val_labels)


def run_preprocessing_grid_search(
    training_character_docs: Dict[str, str],
    validation_character_docs: Dict[str, str],
    config_options: Dict[str, List[Any]],
) -> pd.DataFrame:
    """
    Execute a systematic grid search over preprocessing hyperparameters (Question 1).
    """
    keys = list(config_options.keys())
    values = list(config_options.values())
    configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    train_labels = sorted(training_character_docs.keys())
    val_labels = sorted(validation_character_docs.keys())
    results = []

    for config in configurations:
        vectorizer = DictVectorizer()
        train_dicts = [
            to_feature_vector_dictionary(pre_process(doc, config), 1)
            for _, doc in sorted(training_character_docs.items())
        ]
        val_dicts = [
            to_feature_vector_dictionary(pre_process(doc, config), 1)
            for _, doc in sorted(validation_character_docs.items())
        ]
        train_mat = vectorizer.fit_transform(train_dicts)
        val_mat = vectorizer.transform(val_dicts)
        mean_rank, _, acc, _ = compute_IR_evaluation_scores(
            train_mat, val_mat, train_labels, val_labels
        )
        row = {"mean_rank": mean_rank, "accuracy": acc, **config}
        results.append(row)

    results_df = pd.DataFrame(results)
    return results_df.sort_values(by=["mean_rank", "accuracy"], ascending=[True, False])


def analyze_feature_contribution(
    val_name: str,
    train_name: str,
    val_matrix: csr_matrix,
    train_matrix: csr_matrix,
    val_labels: List[str],
    train_labels: List[str],
    vectorizers: Dict[str, Any],
    filter_eol: bool = True,
    top_k: int = 15,
) -> pd.DataFrame:
    """
    Compute element-wise product contribution $c_i = v_{\\text{val}, i} \\cdot v_{\\text{train}, i}$
    between a validation character vector and a training character vector to explain similarity.
    """
    val_idx = val_labels.index(val_name)
    train_idx = train_labels.index(train_name)

    vec_val = val_matrix[val_idx].toarray().flatten()
    vec_train = train_matrix[train_idx].toarray().flatten()
    contribution = vec_val * vec_train

    word_names = vectorizers["word_dict"].get_feature_names_out()
    char_names = vectorizers["char_tdidf"].get_feature_names_out()
    char_names_tagged = [f"{name} (char)" for name in char_names]
    feature_names = np.concatenate([word_names, char_names_tagged])

    df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Val_Weight": vec_val,
            "Train_Weight": vec_train,
            "Contribution": contribution,
        }
    )
    if filter_eol:
        block_list = {
            "_eol_",
            "eol_ (char)",
            "_eol (char)",
            "_eo (char)",
            "ol_ (char)",
            "ol_  (char)",
            " _eo (char)",
        }
        df = df[~df["Feature"].isin(block_list)]

    return df.sort_values(by="Contribution", ascending=False).head(top_k).reset_index(drop=True)


def analyze_missing_feature_contribution(
    val_name: str,
    train_name: str,
    val_matrix: csr_matrix,
    train_matrix: csr_matrix,
    val_labels: List[str],
    train_labels: List[str],
    vectorizers: Dict[str, Any],
    filter_eol: bool = True,
    top_k: int = 15,
) -> pd.DataFrame:
    """
    Identify high-weight training features that are absent or under-represented in validation
    ($|\\text{Train\\_Weight} - \\text{Val\\_Weight}|$), explaining self-retrieval failures (e.g., Shirley).
    """
    val_idx = val_labels.index(val_name)
    train_idx = train_labels.index(train_name)

    vec_val = val_matrix[val_idx].toarray().flatten()
    vec_train = train_matrix[train_idx].toarray().flatten()
    contribution = vec_val * vec_train

    word_names = vectorizers["word_dict"].get_feature_names_out()
    char_names = vectorizers["char_tdidf"].get_feature_names_out()
    char_names_tagged = [f"{name} (char)" for name in char_names]
    feature_names = np.concatenate([word_names, char_names_tagged])

    df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Val_Weight": vec_val,
            "Train_Weight": vec_train,
            "Contribution": contribution,
        }
    )
    if filter_eol:
        block_list = {
            "_eol_",
            "eol_ (char)",
            "_eol (char)",
            "_eo (char)",
            "ol_ (char)",
            "ol_  (char)",
            " _eo (char)",
        }
        df = df[~df["Feature"].isin(block_list)]

    df["diff"] = np.abs(df["Train_Weight"] - df["Val_Weight"])
    return (
        df[df["Train_Weight"] > 0]
        .sort_values(by="diff", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def plot_heat_map_similarity(
    df: pd.DataFrame, output_path: Optional[str] = None
) -> plt.Figure:
    """Plot the $16 \\times 16$ character cosine similarity matrix heatmap."""
    test_labels = sorted(list(set(df.sort_values(["doc1"])["doc1"])))
    cm = [[1.0] * (len(test_labels) + 2)]
    for target in test_labels:
        new_row = [1.0]
        for x in df.sort_values(["doc1", "doc2"])[df["doc1"] == target]["similarity"]:
            new_row.append(float(x))
        new_row.append(1.0)
        cm.append(new_row)
    cm.append([1.0] * (len(test_labels) + 2))

    labels = [""] + test_labels + [""]
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title("Similarity matrix between documents as vectors")
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)

    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, round(cm[i][j], 3), ha="center", va="center", color="w", fontsize=8)

    plt.xlabel("Training Vector Doc")
    plt.ylabel("Test Vector Doc")
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
    return fig
