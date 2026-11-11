#!/usr/bin/env python
"""
Quick inspection of test predictions from a single LOOCV fold.

Usage:
    python scripts/inspect_fold_predictions.py --fold_dir nas/output/supervised/xcpd_nosmooth_general_fold-101

This prints:
- Summary statistics
- Per-subject accuracy breakdown
- Per-class accuracy breakdown
- Confusion matrix
- List of misclassified samples
"""

import json
import argparse
import numpy as np
from pathlib import Path
from tabulate import tabulate


def load_predictions_and_summary(fold_dir):
    """Load predictions and summary from a fold directory."""
    fold_path = Path(fold_dir) / "test_predictions"
    
    # Find prediction files
    pred_files = list(fold_path.glob("test_predictions_fold_*.json"))
    if not pred_files:
        raise FileNotFoundError(f"No predictions found in {fold_path}")
    
    summary_files = list(fold_path.glob("test_summary_fold_*.json"))
    
    with open(pred_files[0]) as f:
        predictions = json.load(f)
    
    summary = None
    if summary_files:
        with open(summary_files[0]) as f:
            summary = json.load(f)
    
    return predictions, summary


def print_overview(predictions, summary):
    """Print overview statistics."""
    print("\n" + "="*70)
    print("FOLD TEST PREDICTIONS SUMMARY")
    print("="*70)
    
    if summary:
        print(f"\nFold ID:           {summary.get('fold_id', 'unknown')}")
        print(f"Number of samples: {summary.get('num_samples', len(predictions))}")
        print(f"Overall accuracy:  {summary.get('overall_accuracy', 0):.3f}")
        
        if 'per_class_accuracy' in summary:
            print("\nPer-class accuracy:")
            for class_name, acc in summary['per_class_accuracy'].items():
                if acc is not None:
                    print(f"  {class_name:12s}: {acc:.3f}")
                else:
                    print(f"  {class_name:12s}: N/A (no samples)")
        
        if 'confusion_matrix' in summary:
            cm = np.array(summary['confusion_matrix'])
            labels = summary.get('confusion_matrix_labels', ['class_0', 'class_1', 'class_2'])
            print(f"\nConfusion Matrix (rows=true, cols=predicted):")
            print(f"{'':12s}", end="")
            for label in labels:
                print(f"{label:>10s}", end="")
            print()
            for i, label in enumerate(labels):
                print(f"{label:>12s}", end="")
                for j in range(len(labels)):
                    print(f"{cm[i, j]:>10d}", end="")
                print()


def print_per_subject_breakdown(summary):
    """Print per-subject accuracy."""
    if not summary or 'per_subject_accuracy' not in summary:
        return
    
    print("\nPer-subject accuracy:")
    per_subj = summary['per_subject_accuracy']
    
    table_data = []
    for subj in sorted(per_subj.keys()):
        acc = per_subj[subj]['accuracy']
        num_samples = per_subj[subj]['num_samples']
        table_data.append([subj, f"{acc:.3f}", num_samples])
    
    print(tabulate(table_data, headers=['Subject', 'Accuracy', 'N Samples'], tablefmt='simple'))


def print_misclassifications(predictions, top_n=20):
    """Print top misclassified samples."""
    class_names = {0: 'tennis', 1: 'navigation', 2: 'squeeze'}
    
    misclassified = [p for p in predictions if not p['correct']]
    if not misclassified:
        print("\nNo misclassifications!")
        return
    
    # Sort by prediction confidence (highest softmax prob for wrong class)
    misclassified_sorted = sorted(
        misclassified,
        key=lambda p: max(p['softmax_probs']),
        reverse=True
    )
    
    print(f"\nTop {min(top_n, len(misclassified))} misclassifications "
          f"(sorted by confidence):")
    print(f"{'':5s} {'Subject':10s} {'True':12s} {'Predicted':12s} {'Confidence':12s}")
    print("-" * 55)
    
    for i, p in enumerate(misclassified_sorted[:top_n]):
        true_name = p['true_label_name']
        pred_name = p['predicted_label_name']
        conf = max(p['softmax_probs'])
        print(f"{i+1:>3d}. {p['subject']:>10s} {true_name:>12s} {pred_name:>12s} {conf:>12.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='Inspect test predictions from a single LOOCV fold'
    )
    parser.add_argument('--fold_dir', type=str, required=True,
                        help='Path to fold output directory (e.g., '
                             'nas/output/supervised/xcpd_nosmooth_general_fold-101)')
    parser.add_argument('--misclass_top_n', type=int, default=20,
                        help='Number of top misclassifications to show (default: 20)')
    
    args = parser.parse_args()
    
    try:
        predictions, summary = load_predictions_and_summary(args.fold_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    print_overview(predictions, summary)
    print_per_subject_breakdown(summary)
    print_misclassifications(predictions, top_n=args.misclass_top_n)
    print("\n" + "="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    try:
        exit(main())
    except ImportError as e:
        if 'tabulate' in str(e):
            print("Note: Install tabulate for better formatting: pip install tabulate")
        # Fall back to simpler printing
        parser = argparse.ArgumentParser()
        parser.add_argument('--fold_dir', type=str, required=True)
        parser.add_argument('--misclass_top_n', type=int, default=20)
        args = parser.parse_args()
        
        predictions, summary = load_predictions_and_summary(args.fold_dir)
        print_overview(predictions, summary)
        print_misclassifications(predictions, top_n=args.misclass_top_n)
