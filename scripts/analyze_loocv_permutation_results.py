#!/usr/bin/env python
"""
Analyze LOOCV Permutation Test Results: Compare Real vs. Randomized Labels

After training models on both real and randomized labels via
loocv_permutation_testing.sh, this script aggregates results and
computes proper permutation-based null distributions.

Usage:
    python scripts/analyze_loocv_permutation_results.py \
        --real_dir "nas/output/supervised/xcpd_nosmooth_general_fold-*" \
        --perm_dir "nas/output/supervised/xcpd_nosmooth_general_fold-*_perm*" \
        --n_permutations 3

Output:
    - loocv_permutation_results.json (per-fold real vs. perm accuracies)
    - loocv_permutation_comparison.png (visualization)
    - Console summary with p-values and effect sizes
"""

import json
import argparse
import numpy as np
from pathlib import Path
from glob import glob
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plots will not be generated.")


def load_fold_summary(output_dir):
    """Load test_summary_fold_X.json from fold output directory."""
    # Try lightning_logs structure first (new PyTorch Lightning format)
    fold_path = Path(output_dir) / "lightning_logs" / "version_0" / "test_predictions"
    if not fold_path.exists():
        # Fall back to direct test_predictions (old format)
        fold_path = Path(output_dir) / "test_predictions"
    
    summary_files = list(fold_path.glob("test_summary_fold_*.json"))
    if not summary_files:
        return None
    
    with open(summary_files[0]) as f:
        summary = json.load(f)
    
    return summary


def load_per_sample_predictions(output_dir):
    """Load per-sample predictions from test_predictions_fold_X.json."""
    # Try lightning_logs structure first (new PyTorch Lightning format)
    fold_path = Path(output_dir) / "lightning_logs" / "version_0" / "test_predictions"
    if not fold_path.exists():
        # Fall back to direct test_predictions (old format)
        fold_path = Path(output_dir) / "test_predictions"
    
    pred_files = list(fold_path.glob("test_predictions_fold_*.json"))
    if not pred_files:
        return None
    
    with open(pred_files[0]) as f:
        predictions = json.load(f)
    
    return predictions


def extract_fold_id(project_name):
    """Extract fold ID from project name.
    Examples: 
    - 'xcpd_nosmooth_general_fold-101' -> 101
    - 'xcpd_nosmooth_general_fold-101_perm1' -> (101, 1)
    """
    try:
        fold_part = project_name.split("fold-")[-1]
        if "_perm" in fold_part:
            fold_id, perm_part = fold_part.split("_perm")
            return int(fold_id), int(perm_part)
        else:
            return int(fold_part), None
    except (ValueError, IndexError):
        return None, None


def aggregate_results(real_pattern, perm_pattern, n_permutations):
    """
    Load accuracy results for real and permuted models.
    
    Returns:
        dict mapping fold_id -> {
            'real_accuracy': float,
            'real_per_sample': [0/1] correctness per sample,
            'perm_accuracies': [float] of length n_permutations,
            'perm_per_sample': {perm_id: [0/1]} per-sample correctness,
        }
    """
    results = {}
    
    # Load real model accuracies and per-sample predictions
    print(f"Loading real model results from: {real_pattern}")
    real_dirs = sorted(glob(real_pattern))
    for real_dir in real_dirs:
        path = Path(real_dir)
        project_name = path.name
        fold_id, perm_id = extract_fold_id(project_name)
        
        if fold_id is None or perm_id is not None:
            continue  # Skip if not a real model
        
        summary = load_fold_summary(real_dir)
        if summary is None:
            print(f"  Warning: No summary found for {project_name}")
            continue
        
        real_acc = summary.get('overall_accuracy')
        if real_acc is None:
            print(f"  Warning: No accuracy in summary for {project_name}")
            continue
        
        # Load per-sample predictions
        per_sample_preds = load_per_sample_predictions(real_dir)
        per_sample_correct = []
        if per_sample_preds is not None:
            per_sample_correct = [p['correct'] for p in per_sample_preds]
        
        results[fold_id] = {
            'real_accuracy': float(real_acc),
            'real_per_sample': per_sample_correct,
            'perm_accuracies': [],
            'perm_per_sample': {},
        }
        print(f"  fold {fold_id}: real accuracy {real_acc:.3f} ({len(per_sample_correct)} samples)")
    
    # Load permuted model accuracies and per-sample predictions
    print(f"\nLoading permuted model results from: {perm_pattern}")
    perm_dirs = sorted(glob(perm_pattern))
    for perm_dir in perm_dirs:
        path = Path(perm_dir)
        project_name = path.name
        fold_id, perm_id = extract_fold_id(project_name)
        
        if fold_id is None or perm_id is None:
            continue  # Skip if not a permutation model
        
        if fold_id not in results:
            print(f"  Warning: No real model for fold {fold_id}, skipping perm{perm_id}")
            continue
        
        summary = load_fold_summary(perm_dir)
        if summary is None:
            print(f"  Warning: No summary found for fold {fold_id} perm {perm_id}")
            continue
        
        perm_acc = summary.get('overall_accuracy')
        if perm_acc is None:
            print(f"  Warning: No accuracy in summary for fold {fold_id} perm {perm_id}")
            continue
        
        # Load per-sample predictions
        per_sample_preds = load_per_sample_predictions(perm_dir)
        per_sample_correct = []
        if per_sample_preds is not None:
            per_sample_correct = [p['correct'] for p in per_sample_preds]
        
        results[fold_id]['perm_accuracies'].append(float(perm_acc))
        results[fold_id]['perm_per_sample'][perm_id] = per_sample_correct
        print(f"  fold {fold_id}, perm{perm_id}: {perm_acc:.3f} ({len(per_sample_correct)} samples)")
    
    
    # Validate: all folds have n_permutations permutation accuracies
    valid_results = {}
    for fold_id, data in results.items():
        if len(data['perm_accuracies']) != n_permutations:
            print(f"  Warning: fold {fold_id} has {len(data['perm_accuracies'])} "
                  f"permutations (expected {n_permutations}), skipping")
            continue
        valid_results[fold_id] = data
    
    return valid_results


def compute_statistics(results):
    """Compute per-fold and aggregate statistics."""
    fold_ids = sorted(results.keys())
    
    stats = {}
    for fold_id in fold_ids:
        data = results[fold_id]
        real_acc = data['real_accuracy']
        perm_accs = np.array(data['perm_accuracies'])
        
        # P-value: fraction of permutations >= real accuracy
        # (one-tailed test: is real accuracy higher than permutation?)
        p_value = (perm_accs >= real_acc).sum() / len(perm_accs)
        
        # Effect size: how much better is real than average permutation?
        effect_size = real_acc - perm_accs.mean()
        effect_size_std = (real_acc - perm_accs.mean()) / (perm_accs.std() + 1e-8)
        
        stats[fold_id] = {
            'real_accuracy': real_acc,
            'perm_mean': float(perm_accs.mean()),
            'perm_std': float(perm_accs.std()),
            'perm_min': float(perm_accs.min()),
            'perm_max': float(perm_accs.max()),
            'effect_size': float(effect_size),
            'effect_size_std': float(effect_size_std),
            'p_value': float(p_value),
            'significant_p05': bool(p_value < 0.05),
        }
    
    return stats


def print_summary(stats):
    """Print summary statistics."""
    fold_ids = sorted(stats.keys())
    
    print("\n" + "="*80)
    print("LOOCV PERMUTATION TEST RESULTS (Real Labels vs. Randomized Labels)")
    print("="*80)
    
    real_accs = [stats[fid]['real_accuracy'] for fid in fold_ids]
    perm_means = [stats[fid]['perm_mean'] for fid in fold_ids]
    p_values = [stats[fid]['p_value'] for fid in fold_ids]
    effect_sizes = [stats[fid]['effect_size'] for fid in fold_ids]
    
    print(f"\nAcross {len(fold_ids)} folds:")
    print(f"  Real accuracy (mean ± SEM):      {np.mean(real_accs):.3f} ± {np.std(real_accs)/np.sqrt(len(real_accs)):.3f}")
    print(f"  Real accuracy range:             {np.min(real_accs):.3f} - {np.max(real_accs):.3f}")
    print(f"  Randomized label accuracy (mean): {np.mean(perm_means):.3f} ± {np.std(perm_means):.3f}")
    print(f"  Mean improvement (real - random): {np.mean(effect_sizes):.3f} ± {np.std(effect_sizes):.3f}")
    print(f"\n  Folds significant at p < 0.05:   {sum(stats[fid]['significant_p05'] for fid in fold_ids)} / {len(fold_ids)}")
    print(f"  Median p-value:                   {np.median(p_values):.3f}")
    print(f"  Mean effect size (std units):     {np.mean([stats[fid]['effect_size_std'] for fid in fold_ids]):.2f}")
    print("="*80 + "\n")
    
    # Per-fold details
    print("Per-fold breakdown:")
    print(f"{'Fold':>6s} {'Real Acc':>10s} {'Perm Mean':>10s} {'Improvement':>12s} {'P-value':>10s} {'Sig':>5s}")
    print("-" * 60)
    for fold_id in fold_ids:
        s = stats[fold_id]
        sig = "✓" if s['significant_p05'] else ""
        print(f"{fold_id:>6d} {s['real_accuracy']:>10.3f} {s['perm_mean']:>10.3f} "
              f"{s['effect_size']:>12.3f} {s['p_value']:>10.3f} {sig:>5s}")


def save_results(stats, output_file='loocv_permutation_results.json'):
    """Save detailed results to JSON."""
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved results to {output_file}")


def plot_results(stats, results, output_file='loocv_permutation_comparison.png'):
    """Generate comparison plots: bar chart and KDE distribution."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot generation (matplotlib not installed)")
        return
    
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        print("Warning: scipy not installed, skipping KDE plot")
        scipy_stats = None
    
    fold_ids = sorted(stats.keys())
    real_accs = [stats[fid]['real_accuracy'] for fid in fold_ids]
    perm_means = [stats[fid]['perm_mean'] for fid in fold_ids]
    perm_stds = [stats[fid]['perm_std'] for fid in fold_ids]
    
    # Collect all per-sample accuracies
    real_per_sample_all = []
    perm_per_sample_all = []
    
    for fold_id in fold_ids:
        if fold_id in results:
            real_per_sample_all.extend(results[fold_id]['real_per_sample'])
            for perm_id, perm_samples in results[fold_id]['perm_per_sample'].items():
                perm_per_sample_all.extend(perm_samples)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: Per-fold bar chart (existing)
    ax = axes[0]
    x = np.arange(len(fold_ids))
    width = 0.35
    
    ax.bar(x - width/2, real_accs, width, label='Real Labels', 
           color='steelblue', alpha=0.8)
    ax.errorbar(x + width/2, perm_means, yerr=perm_stds, fmt='o', 
                color='coral', ecolor='darkred', capsize=5, capthick=2,
                label='Randomized Labels (mean ± std)', markersize=6, alpha=0.8)
    
    ax.set_xlabel('LOOCV Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Per-Fold Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{fid}' for fid in fold_ids], rotation=45, fontsize=9)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Panel 2: Per-sample KDE distributions
    ax = axes[1]
    
    if scipy_stats is not None and len(real_per_sample_all) > 0 and len(perm_per_sample_all) > 0:
        # Add small jitter for visualization
        real_samples_jittered = np.array(real_per_sample_all) + np.random.normal(0, 0.01, len(real_per_sample_all))
        perm_samples_jittered = np.array(perm_per_sample_all) + np.random.normal(0, 0.01, len(perm_per_sample_all))
        
        # Plot KDE
        from scipy.stats import gaussian_kde
        
        real_kde = gaussian_kde(real_samples_jittered, bw_method=0.1)
        perm_kde = gaussian_kde(perm_samples_jittered, bw_method=0.1)
        
        x_range = np.linspace(-0.1, 1.1, 200)
        
        ax.plot(x_range, real_kde(x_range), color='steelblue', linewidth=3, 
                label=f'Real Labels (n={len(real_per_sample_all)} samples)')
        ax.fill_between(x_range, real_kde(x_range), alpha=0.2, color='steelblue')
        
        ax.plot(x_range, perm_kde(x_range), color='coral', linewidth=3, 
                label=f'Randomized Labels (n={len(perm_per_sample_all)} samples)')
        ax.fill_between(x_range, perm_kde(x_range), alpha=0.2, color='coral')
        
        ax.set_xlabel('Per-Sample Accuracy (0=incorrect, 1=correct)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title('Per-Sample Accuracy Distribution (KDE)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='upper center')
        ax.set_xlim([-0.1, 1.1])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add statistics text
        real_mean = np.mean(real_per_sample_all)
        perm_mean = np.mean(perm_per_sample_all)
        real_std = np.std(real_per_sample_all)
        perm_std = np.std(perm_per_sample_all)
        
        stats_text = f"Real: μ={real_mean:.3f}, σ={real_std:.3f}\nRandom: μ={perm_mean:.3f}, σ={perm_std:.3f}"
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'Per-sample predictions not available\nor insufficient samples for KDE', 
                transform=ax.transAxes, fontsize=12, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze LOOCV permutation test results'
    )
    parser.add_argument('--real_dir', type=str, required=True,
                        help='Glob pattern for real model output dirs')
    parser.add_argument('--perm_dir', type=str, required=True,
                        help='Glob pattern for permutation model output dirs')
    parser.add_argument('--n_permutations', type=int, required=True,
                        help='Number of permutations per fold')
    parser.add_argument('--output_file', type=str, default='loocv_permutation_results.json',
                        help='Output file for results')
    parser.add_argument('--plot_file', type=str, default='loocv_permutation_comparison.png',
                        help='Output file for plot')
    parser.add_argument('--no_plot', action='store_true',
                        help='Skip plot generation')
    
    args = parser.parse_args()
    
    print("Aggregating LOOCV permutation test results...")
    
    try:
        results = aggregate_results(args.real_dir, args.perm_dir, args.n_permutations)
    except Exception as e:
        print(f"Error loading results: {e}")
        return 1
    
    if not results:
        print("No results found. Check glob patterns and directory structure.")
        return 1
    
    stats = compute_statistics(results)
    print_summary(stats)
    save_results(stats, args.output_file)
    
    if not args.no_plot:
        plot_results(stats, results, args.plot_file)
    
    return 0


if __name__ == '__main__':
    exit(main())
