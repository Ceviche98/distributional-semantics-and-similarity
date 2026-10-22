#!/usr/bin/env python3
"""
CLI script to run systematic grid searches over preprocessing (Question 1)
and multi-view feature weighting (Question 2).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
from vector_semantics import (
    create_character_document_from_dataframe,
    load_script_dataset,
    run_config_features,
    run_preprocessing_grid_search,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing or feature weighting grid search.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing CSV datasets")
    parser.add_argument("--stage", choices=["q1-fast", "q1-full", "q2-weights"], default="q2-weights")
    parser.add_argument("--output", type=str, default=None, help="Optional CSV path to save grid search results")
    args = parser.parse_args()

    train_df = load_script_dataset(os.path.join(args.data_dir, "training.csv"))
    val_df = load_script_dataset(os.path.join(args.data_dir, "val.csv"))

    train_docs = create_character_document_from_dataframe(train_df, max_line_count=300)
    val_docs = create_character_document_from_dataframe(val_df, max_line_count=50)

    if args.stage in ("q1-fast", "q1-full"):
        if args.stage == "q1-full":
            grid = {
                "lower_case": [True, False],
                "erase_numbers": [True, False],
                "erase_ellipsis": [True, False],
                "contractions": [True, False],
                "erase_stop_words": [True, False],
                "erase_stop_words_negation": [True, False],
                "punctuation": [True, False],
                "split": ["split", "tokenizer"],
                "lex_norm": ["none", "stemming", "lemmatization"],
            }
        else:
            grid = {
                "lower_case": [True],
                "erase_numbers": [True],
                "erase_ellipsis": [True, False],
                "contractions": [True],
                "erase_stop_words": [True],
                "erase_stop_words_negation": [True],
                "punctuation": [True],
                "split": ["split", "tokenizer"],
                "lex_norm": ["none", "stemming", "lemmatization"],
            }
        df_res = run_preprocessing_grid_search(train_docs, val_docs, grid)
    else:
        rows = []
        for w_uni in [0.8, 1.0, 1.25]:
            for w_char in [0.7, 0.8, 1.0]:
                for sublinear in [True, False]:
                    feat_cfg = {
                        "ngrams": 1,
                        "td-idf": True,
                        "char_min_df": 3,
                        "charmin_range": 4,
                        "charmax_range": 4,
                        "w_uni": w_uni,
                        "w_char": w_char,
                        "sublinear_tf": sublinear,
                    }
                    rank, sim, acc, _ = run_config_features(
                        train_docs, val_docs, config_features=feat_cfg
                    )
                    rows.append(
                        {
                            "mean_rank": rank,
                            "accuracy": acc,
                            "mean_cosine_sim": sim,
                            **feat_cfg,
                        }
                    )
        df_res = pd.DataFrame(rows).sort_values(
            by=["mean_rank", "accuracy"], ascending=[True, False]
        )

    print("=== Top 10 Configurations ===")
    print(df_res.head(10).to_string(index=False))
    if args.output:
        df_res.to_csv(args.output, index=False)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
