#!/usr/bin/env python3
"""
CLI script to evaluate character vector similarity, retrieve character rankings,
and perform feature-level error analysis on validation and blind test splits.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from vector_semantics import (
    analyze_feature_contribution,
    analyze_missing_feature_contribution,
    compute_IR_evaluation_scores,
    create_character_document_from_dataframe,
    create_document_matrix_from_corpus,
    load_script_dataset,
    plot_heat_map_similarity,
    pre_process_baseline,
    to_feature_vector_dictionary,
)
from sklearn.feature_extraction import DictVectorizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate EastEnders character distributional similarity on validation and test sets."
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing training.csv, val.csv, test.csv")
    parser.add_argument("--mode", choices=["optimal", "baseline"], default="optimal", help="Representation mode")
    parser.add_argument("--error-analysis", action="store_true", help="Print top contributing & missing features for key pairs")
    parser.add_argument("--save-heatmap", type=str, default=None, help="Optional path to save similarity heatmap PNG")
    args = parser.parse_args()

    train_df = load_script_dataset(os.path.join(args.data_dir, "training.csv"))
    val_df = load_script_dataset(os.path.join(args.data_dir, "val.csv"))
    test_df = load_script_dataset(os.path.join(args.data_dir, "test.csv"))

    train_docs = create_character_document_from_dataframe(train_df, max_line_count=300)
    val_docs = create_character_document_from_dataframe(val_df, max_line_count=50)
    test_docs = create_character_document_from_dataframe(test_df, max_line_count=50)

    train_labels = sorted(train_docs.keys())
    val_labels = sorted(val_docs.keys())
    test_labels = sorted(test_docs.keys())

    if args.mode == "baseline":
        vec = DictVectorizer()
        train_mat = vec.fit_transform(
            [to_feature_vector_dictionary(pre_process_baseline(train_docs[k]), 1) for k in train_labels]
        )
        val_mat = vec.transform(
            [to_feature_vector_dictionary(pre_process_baseline(val_docs[k]), 1) for k in val_labels]
        )
        test_mat = vec.transform(
            [to_feature_vector_dictionary(pre_process_baseline(test_docs[k]), 1) for k in test_labels]
        )
        vectorizers = {"word_dict": vec}
    else:
        vectorizers = {}
        train_mat = create_document_matrix_from_corpus(train_docs, vectorizers, fitting=True)
        val_mat = create_document_matrix_from_corpus(val_docs, vectorizers, fitting=False)
        test_mat = create_document_matrix_from_corpus(test_docs, vectorizers, fitting=False)

    val_rank, val_sim, val_acc, val_sim_df = compute_IR_evaluation_scores(
        train_mat, val_mat, train_labels, val_labels
    )
    test_rank, test_sim, test_acc, test_sim_df = compute_IR_evaluation_scores(
        train_mat, test_mat, train_labels, test_labels
    )

    print(f"=== Evaluation Mode: {args.mode.upper()} ===")
    print(f"[Validation] Mean Rank: {val_rank:.4f} | Mean Cosine Sim: {val_sim:.4f} | Accuracy: {val_acc:.4f} ({int(val_acc*16)}/16)")
    print(f"[Blind Test] Mean Rank: {test_rank:.4f} | Mean Cosine Sim: {test_sim:.4f} | Accuracy: {test_acc:.4f} ({int(test_acc*16)}/16)")

    if args.error_analysis and args.mode == "optimal":
        print("\n=== Linguistic Feature Attribution (SHIRLEY Val vs SEAN Train) ===")
        print(
            analyze_feature_contribution(
                "SHIRLEY", "SEAN", val_mat, train_mat, val_labels, train_labels, vectorizers
            ).to_string(index=False)
        )
        print("\n=== Missing Anchor Analysis (SHIRLEY Val vs SHIRLEY Train) ===")
        print(
            analyze_missing_feature_contribution(
                "SHIRLEY", "SHIRLEY", val_mat, train_mat, val_labels, train_labels, vectorizers
            ).to_string(index=False)
        )

    if args.save_heatmap:
        plot_heat_map_similarity(val_sim_df, output_path=args.save_heatmap)
        print(f"Saved similarity heatmap to {args.save_heatmap}")


if __name__ == "__main__":
    main()
