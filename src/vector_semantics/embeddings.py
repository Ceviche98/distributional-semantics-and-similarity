"""
Dense syntactic (POS) feature extraction, neural document embeddings (`SentenceTransformer`),
and multi-matrix weighted sparse-dense ensembling.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from nltk.tokenize import word_tokenize
import numpy as np
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.preprocessing import normalize

FIRST_PERSON_PRONOUNS = {
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
}

SECOND_PERSON_PRONOUNS = {
    "you", "your", "yours", "yourself", "yourselves",
}


def get_pos_features(
    text: str,
    tokens: List[str],
    pos_tagger: Optional[ object ] = None,
) -> Tuple[float, ...]:
    """
    Extract 10 dense syntactic and stylistic ratios per character document (Question 2 Section 5):
    1. Average words per sentence (`avg_word_per_sentence`)
    2. Question mark frequency per sentence (`num_question_mark / total_sentences`)
    3. Interrogative sentence ratio (`num_sentences_end_question / total_sentences`)
    4. Adjective rate (`num_adjectives / total_sentences`)
    5. Verb rate (`num_verbs / total_sentences`)
    6. Noun rate (`num_nouns / total_sentences`)
    7. Pronoun rate (`num_pronouns / total_sentences`)
    8. Adjective-to-noun ratio (`adj_noun_ratio`)
    9. First-person pronoun rate (`num_first_person / total_sentences`)
    10. Second-person pronoun rate (`num_second_person / total_sentences`)
    """
    sentences = [s for s in re.split(r"\s*_eol_\s*", text, flags=re.IGNORECASE) if s.strip()]
    total_sentences = max(len(sentences), 1)

    if pos_tagger is not None:
        doc = pos_tagger(text)
        num_adjectives = sum(1 for token in doc if token.pos_ == "ADJ")
        num_verbs = sum(1 for token in doc if token.pos_ == "VERB")
        num_nouns = sum(1 for token in doc if token.pos_ == "NOUN")
        num_pronouns = sum(1 for token in doc if token.pos_ == "PRON")
    else:
        import nltk
        try:
            tagged = nltk.pos_tag(tokens)
        except LookupError:
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            tagged = nltk.pos_tag(tokens)
        num_adjectives = sum(1 for _, tag in tagged if tag.startswith("JJ"))
        num_verbs = sum(1 for _, tag in tagged if tag.startswith("VB"))
        num_nouns = sum(1 for _, tag in tagged if tag.startswith("NN"))
        num_pronouns = sum(1 for _, tag in tagged if tag.startswith("PRP") or tag.startswith("WP"))

    num_question_mark = sum(1 for token in tokens if token == "?")
    num_sentences_end_question = sum(1 for s in sentences if s.strip().endswith("?"))

    words_per_sentence = []
    for sentence in sentences:
        try:
            words_per_sentence.append(len(word_tokenize(sentence)))
        except LookupError:
            words_per_sentence.append(len(sentence.split()))
    avg_word_per_sentence = sum(words_per_sentence) / total_sentences

    lower_tokens = [t.lower() for t in tokens]
    num_first_person = sum(1 for t in lower_tokens if t in FIRST_PERSON_PRONOUNS)
    num_second_person = sum(1 for t in lower_tokens if t in SECOND_PERSON_PRONOUNS)
    adj_noun_ratio = num_adjectives / num_nouns if num_nouns > 0 else 0.0

    return (
        float(avg_word_per_sentence),
        float(num_question_mark / total_sentences),
        float(num_sentences_end_question / total_sentences),
        float(num_adjectives / total_sentences),
        float(num_verbs / total_sentences),
        float(num_nouns / total_sentences),
        float(num_pronouns / total_sentences),
        float(adj_noun_ratio),
        float(num_first_person / total_sentences),
        float(num_second_person / total_sentences),
    )


def extract_pos_matrix(
    raw_text_list: Sequence[str],
    token_list: Sequence[List[str]],
    pos_tagger: Optional[object] = None,
) -> csr_matrix:
    """Build a sparse $N \\times 10$ matrix of syntactic/POS features across character documents."""
    rows = [
        list(get_pos_features(text, tokens, pos_tagger=pos_tagger))
        for text, tokens in zip(raw_text_list, token_list)
    ]
    return csr_matrix(np.asarray(rows, dtype=float))


def get_avg_document_embedding(
    raw_text_list: Iterable[str],
    model_name: str = "all-mpnet-base-v2",
    embedding_model: Optional[object] = None,
) -> np.ndarray:
    """
    Compute mean-pooled neural sentence embeddings per character document
    by splitting on `_EOL_` markers and encoding each utterance via `SentenceTransformer`.
    """
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(model_name)

    doc_embeddings = []
    for text in raw_text_list:
        sentences = [s.strip() for s in re.split(r"_eol_", text, flags=re.IGNORECASE) if s.strip()]
        if not sentences:
            sentences = [text]
        sent_embeddings = embedding_model.encode(sentences)
        avg_embedding = np.mean(sent_embeddings, axis=0)
        doc_embeddings.append(avg_embedding)
    return np.asarray(doc_embeddings, dtype=float)


def combine_weighted_matrices(
    matrices: Sequence[Union[csr_matrix, np.ndarray]],
    weights: Sequence[float],
    l2_normalize_each: bool = True,
) -> csr_matrix:
    """
    Normalize individual feature spaces to unit L2 norm, scale each block by scalar weight $w_i$,
    and horizontally stack them into a unified hybrid representation:
    $$\\mathbf{v}_{\\text{hybrid}} = \\left[ w_1 \\hat{\\mathbf{v}}^{(1)} \\;\\|\\; w_2 \\hat{\\mathbf{v}}^{(2)} \\;\\|\\; \\dots \\right]$$
    """
    blocks = []
    for mat, w in zip(matrices, weights):
        sp_mat = mat if issparse(mat) else csr_matrix(np.asarray(mat, dtype=float))
        if l2_normalize_each:
            sp_mat = normalize(sp_mat, norm="l2")
        blocks.append(sp_mat * float(w))
    return hstack(blocks).tocsr()


def hybrid_score_ensemble(
    sparse_sim_matrix: np.ndarray,
    dense_sim_matrix: np.ndarray,
    alpha: float = 0.85,
) -> np.ndarray:
    """
    Convex combination of sparse lexical/subword similarity scores and dense neural similarity scores:
    $$S_{\\text{ensemble}} = \\alpha S_{\\text{sparse}} + (1 - \\alpha) S_{\\text{dense}}$$
    """
    return alpha * np.asarray(sparse_sim_matrix, dtype=float) + (1.0 - alpha) * np.asarray(dense_sim_matrix, dtype=float)
