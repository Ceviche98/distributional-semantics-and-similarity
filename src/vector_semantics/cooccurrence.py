"""
Sparse document-term, character n-gram, and sliding-window word-context co-occurrence matrix construction.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from scipy.sparse import csr_matrix, lil_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from .corpus import build_vocabulary


def generate_ngrams(tokens: List[str], n: int) -> List[str]:
    """
    Generate word n-grams joined by underscores from a token sequence.
    Example: `(['not', 'good'], 2) -> ['not_good']`
    """
    if n == 1:
        return list(tokens)
    return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def get_all_ngrams(tokens: List[str], max_n: int, include_unigrams: bool = True) -> List[str]:
    """Generate all n-grams up to `max_n`."""
    all_ngrams: List[str] = []
    start_n = 1 if include_unigrams else 2
    for n in range(start_n, max_n + 1):
        all_ngrams.extend(generate_ngrams(tokens, n))
    return all_ngrams


def to_feature_vector_dictionary(
    character_doc: List[str],
    ngram: int = 1,
    extra_features: Optional[List[str]] = None,
    include_unigrams: bool = True,
) -> Dict[str, int]:
    """
    Convert a preprocessed token list into a frequency dictionary over unigrams or higher-order n-grams.
    """
    if ngram <= 1:
        counts = Counter(character_doc)
    else:
        counts = Counter(
            get_all_ngrams(character_doc, ngram, include_unigrams=include_unigrams)
        )
    result = dict(counts)
    if extra_features:
        for feat in extra_features:
            result[feat] = result.get(feat, 0) + 1
    return result


def build_cooccurrence_matrix(
    tokenized_docs: Iterable[List[str]],
    window_size: int = 2,
    vocab: Optional[Dict[str, int]] = None,
    min_freq: int = 1,
    symmetric: bool = True,
    distance_weighting: bool = False,
) -> Tuple[csr_matrix, Dict[str, int]]:
    """
    Construct a sparse $|V| \\times |V|$ word-context co-occurrence matrix $C$
    using a sliding context window of radius $w = \\text{window\\_size}$.

    Parameters
    ----------
    tokenized_docs : Iterable[List[str]]
        Tokenized documents or utterances.
    window_size : int
        Context window radius $w \\ge 1$ around target word $w_t$.
    vocab : Optional[Dict[str, int]]
        Pre-built vocabulary mapping `word -> index`. If None, built from `tokenized_docs`.
    min_freq : int
        Minimum token frequency if building vocabulary.
    symmetric : bool
        If True, counts both left and right contexts symmetrically. If False, counts only right context (directional).
    distance_weighting : bool
        If True, weights context occurrences by inverse distance $1 / d$.
    """
    docs_list = list(tokenized_docs)
    if vocab is None:
        vocab, _ = build_vocabulary(docs_list, min_freq=min_freq)

    vocab_size = len(vocab)
    matrix = lil_matrix((vocab_size, vocab_size), dtype=float)

    for doc in docs_list:
        n_tokens = len(doc)
        for i, target in enumerate(doc):
            target_idx = vocab.get(target)
            if target_idx is None:
                continue

            start = max(0, i - window_size) if symmetric else i + 1
            end = min(n_tokens, i + window_size + 1)

            for j in range(start, end):
                if i == j:
                    continue
                context_word = doc[j]
                context_idx = vocab.get(context_word)
                if context_idx is None:
                    continue
                weight = 1.0 / abs(i - j) if distance_weighting else 1.0
                matrix[target_idx, context_idx] += weight

    return matrix.tocsr(), vocab


def build_word_and_char_matrices(
    preprocessed_docs: List[List[str]],
    raw_docs: List[str],
    vectorizers: Dict[str, Any],
    config_features: Dict[str, Any],
    fitting: bool = False,
) -> Tuple[csr_matrix, csr_matrix]:
    """
    Build the dual-channel sparse matrices (word unigram/n-gram matrix + character n-gram matrix).
    """
    ngrams = config_features.get("ngrams", 1)
    dicts_ngrams = [to_feature_vector_dictionary(doc, ngrams) for doc in preprocessed_docs]

    if fitting:
        vectorizers["word_dict"] = DictVectorizer()
        vectorizers["word_dict"].fit(dicts_ngrams)

    matrix_word = vectorizers["word_dict"].transform(dicts_ngrams)

    char_min_range = config_features.get("charmin_range", 4)
    char_max_range = config_features.get("charmax_range", 4)
    char_min_df = config_features.get("char_min_df", 3)
    sublinear_tf = config_features.get("sublinear_tf", True)

    if fitting:
        vectorizers["char_tdidf"] = TfidfVectorizer(
            analyzer="char",
            ngram_range=(char_min_range, char_max_range),
            min_df=char_min_df,
            sublinear_tf=sublinear_tf,
        )
        vectorizers["char_tdidf"].fit(raw_docs)

    matrix_char = vectorizers["char_tdidf"].transform(raw_docs)
    return matrix_word, matrix_char
