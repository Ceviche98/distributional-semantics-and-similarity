"""
Corpus loading, character document aggregation, tokenization, and lexical normalization.
"""

from __future__ import annotations

from collections import Counter
import math
import re
import string
from typing import Any, Dict, Iterable, List, Optional, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd


def _ensure_nltk_resources() -> None:
    """Ensure required NLTK tokenizers and corpora are available locally."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


_ensure_nltk_resources()

VADER_NEGATE: List[str] = [
    "aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
    "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
    "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neednt",
    "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "needn't",
    "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
    "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
    "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "never",
]

ORIGINAL_NEGATION_WORDS = {
    "no", "nor", "not", "don", "don't", "ain", "aren", "aren't", "couldn", "couldn't",
    "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't",
    "haven", "haven't", "isn", "isn't", "mightn", "mightn't", "mustn", "mustn't",
    "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't",
    "weren", "weren't", "won", "won't", "wouldn", "wouldn't",
}

NEGATION_WORDS_SET = ORIGINAL_NEGATION_WORDS.union(set(VADER_NEGATE))

CONTRACTION_MAP: Dict[str, str] = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "couldn't": "could not",
    "didn't": "did not", "don't": "do not", "hasn't": "has not", "haven't": "have not",
    "i'm": "i am", "i've": "i have", "isn't": "is not", "it's": "it is",
    "let's": "let us", "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",
    "she's": "she is", "shouldn't": "should not", "that's": "that is", "there's": "there is",
    "they're": "they are", "wasn't": "was not", "we're": "we are", "weren't": "were not",
    "what's": "what is", "where's": "where is", "who's": "who is", "won't": "will not",
    "wouldn't": "would not", "you're": "you are", "you've": "you have",
    "'s": " is",
}

LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()

DEFAULT_PREPROCESS_CONFIG: Dict[str, Any] = {
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


def load_script_dataset(path: str) -> pd.DataFrame:
    """Load tab-delimited EastEnders script data from CSV."""
    return pd.read_csv(path, delimiter="\t", skip_blank_lines=True)


def create_character_document_from_dataframe(
    df: pd.DataFrame, max_line_count: int = 300, verbose: bool = False
) -> Dict[str, str]:
    """
    Aggregate spoken lines per character up to ``max_line_count`` lines,
    joining each utterance with an ``_EOL_`` boundary marker.
    """
    character_docs: Dict[str, str] = {}
    character_line_count: Dict[str, int] = {}

    for line, name in zip(df["Line"], df["Character_name"]):
        if (isinstance(line, float) and math.isnan(line)) or line == "":
            continue
        if name not in character_docs:
            character_docs[name] = ""
            character_line_count[name] = 0
        if character_line_count[name] == max_line_count:
            continue
        character_docs[name] += str(line) + " _EOL_ "
        character_line_count[name] += 1

    if verbose:
        print("lines per character", character_line_count)
    return character_docs


def expand_contractions(
    text: str, contraction_mapping: Optional[Dict[str, str]] = None
) -> str:
    """Expand English contractions while preserving initial capitalization."""
    if contraction_mapping is None:
        contraction_mapping = CONTRACTION_MAP

    contractions_pattern = re.compile(
        "({})".format("|".join(re.escape(k) for k in contraction_mapping.keys())),
        flags=re.IGNORECASE | re.DOTALL,
    )

    def replace_match(match: re.Match) -> str:
        match_text = match.group(0)
        initial_cap = match_text[0].isupper()
        expanded = contraction_mapping.get(match_text.lower(), match_text)
        return expanded.capitalize() if initial_cap else expanded

    return contractions_pattern.sub(replace_match, text)


def pre_process_baseline(character_text: str) -> List[str]:
    """Baseline whitespace tokenizer."""
    return character_text.split()


def pre_process_raw(character_text: str) -> List[str]:
    """
    Lightweight preprocessing for raw n-gram extraction:
    lowercasing, contraction expansion, NLTK tokenization, and punctuation filtering
    (retaining '?' and '!').
    """
    character_text = character_text.lower()
    character_text = expand_contractions(character_text, CONTRACTION_MAP)
    try:
        tokens = word_tokenize(character_text)
    except LookupError:
        tokens = character_text.split()

    punctuation_to_remove = set(string.punctuation) - {"!", "?"}
    return [t for t in tokens if t not in punctuation_to_remove]


def pre_process(
    character_text: str, config: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Configurable 8-stage preprocessing pipeline evaluated in Question 1:
    1. Lowercasing
    2. Contraction expansion
    3. Digit removal
    4. Ellipsis removal / retention (`...` and `…`)
    5. Tokenization (`split` vs NLTK `word_tokenize`)
    6. Stopword removal (with optional VADER negation preservation)
    7. Punctuation removal (retaining stylistic `!` and `?` markers)
    8. Lexical normalization (`lemmatization`, `stemming`, or `None`)
    """
    cfg = {**DEFAULT_PREPROCESS_CONFIG, **(config or {})}

    if cfg.get("lower_case", True):
        character_text = character_text.lower()

    if cfg.get("contractions", True):
        character_text = expand_contractions(character_text, CONTRACTION_MAP)

    if cfg.get("erase_numbers", True):
        character_text = re.sub(r"\d+", "", character_text)

    if cfg.get("erase_ellipsis", False):
        character_text = re.sub(r"\.{3,}|…", " ", character_text)

    if cfg.get("split", "tokenizer") == "tokenizer":
        try:
            tokens = word_tokenize(character_text)
        except LookupError:
            tokens = character_text.split()
    else:
        tokens = character_text.split()

    custom_stop_words_set = set()
    if cfg.get("erase_stop_words", True):
        try:
            stop_words_list = stopwords.words("english")
        except LookupError:
            stop_words_list = []
        if cfg.get("erase_stop_words_negation", True):
            custom_stop_words_set = set(stop_words_list) - NEGATION_WORDS_SET
        else:
            custom_stop_words_set = set(stop_words_list)

    punctuation_to_remove = set(string.punctuation) - {"!", "?"}
    lex_norm = cfg.get("lex_norm", None)

    clean_tokens: List[str] = []
    for token in tokens:
        if cfg.get("punctuation", True) and token in punctuation_to_remove:
            continue
        if cfg.get("erase_stop_words", True) and token in custom_stop_words_set:
            continue
        if lex_norm == "lemmatization":
            token = LEMMATIZER.lemmatize(token)
        elif lex_norm == "stemming":
            token = STEMMER.stem(token)
        clean_tokens.append(token)

    return clean_tokens


def build_vocabulary(
    tokenized_docs: Iterable[List[str]],
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None,
) -> Tuple[Dict[str, int], Counter]:
    """
    Construct a frequency-filtered vocabulary mapping `token -> index` ($|V|$ scaling).
    """
    counts: Counter = Counter()
    for doc in tokenized_docs:
        counts.update(doc)

    filtered = [
        (tok, freq) for tok, freq in counts.items() if freq >= min_freq
    ]
    filtered.sort(key=lambda x: (-x[1], x[0]))

    if max_vocab_size is not None:
        filtered = filtered[:max_vocab_size]

    vocab = {tok: idx for idx, (tok, _) in enumerate(filtered)}
    return vocab, counts
