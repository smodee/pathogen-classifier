"""
Explore and characterize the collected viral genomic sequences.
Generates summary statistics, quality metrics, and visualizations
to inform preprocessing and model architecture decisions.

Usage:
    python src/data_exploration.py
    python src/data_exploration.py --input data/raw/viral_sequences_all.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Plotting imports are deferred so the notebook can control the backend.
# When run as a script, __main__ sets the Agg backend before these are used.
_plt = None
_sns = None

# Ordered by typical genome size (descending) for consistent plot ordering
FAMILY_ORDER = [
    'Poxviridae', 'Coronaviridae', 'Paramyxoviridae',
    'Filoviridae', 'Flaviviridae', 'Arenaviridae', 'Orthomyxoviridae'
]


def _init_plotting():
    """Lazily import matplotlib and seaborn."""
    global _plt, _sns
    if _plt is None:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid", font_scale=1.1)
        _plt = plt
        _sns = sns
    return _plt, _sns


# ---------------------------------------------------------------------------
# Data loading and metrics
# ---------------------------------------------------------------------------

def load_data(path):
    """
    Load the combined viral sequences CSV.

    Args:
        path: Path to viral_sequences_all.csv

    Returns:
        pandas DataFrame with columns: id, description, sequence, length, family
    """
    path = Path(path)
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)

    expected_cols = {'id', 'description', 'sequence', 'length', 'family'}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    print(f"  Loaded {len(df):,} sequences ({df['family'].nunique()} families)")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    return df


def compute_sequence_metrics(df):
    """
    Add derived quality and composition columns.

    Adds: gc_content, n_count, n_percent, is_duplicate
    """
    print("Computing sequence metrics...")

    # GC content (exclude N bases from denominator)
    def _gc(seq):
        seq = seq.upper()
        gc = seq.count('G') + seq.count('C')
        valid = len(seq) - seq.count('N')
        return gc / valid if valid > 0 else 0.0

    df['gc_content'] = df['sequence'].apply(_gc)

    # Recompute N-base stats for correctness
    df['n_count'] = df['sequence'].apply(lambda s: s.upper().count('N'))
    df['n_percent'] = df['n_count'] / df['length']

    # Duplicate detection (exact sequence match)
    df['is_duplicate'] = df.duplicated(subset='sequence', keep='first')

    print(f"  GC content range: {df['gc_content'].min():.3f} – {df['gc_content'].max():.3f}")
    print(f"  Duplicates found: {df['is_duplicate'].sum():,}")
    return df


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary_statistics(df):
    """
    Print and return per-family summary statistics.

    Returns:
        DataFrame with one row per family and summary columns.
    """
    print("\n=== Class Distribution ===")
    counts = df['family'].value_counts()
    total = len(df)
    for fam in FAMILY_ORDER:
        if fam in counts.index:
            n = counts[fam]
            print(f"  {fam:<20s}  {n:>6,}  ({n/total*100:5.1f}%)")
    print(f"  {'TOTAL':<20s}  {total:>6,}")

    print("\n=== Per-Family Statistics ===")
    rows = []
    for fam in FAMILY_ORDER:
        sub = df[df['family'] == fam]
        if sub.empty:
            continue
        rows.append({
            'family': fam,
            'count': len(sub),
            'mean_length': int(sub['length'].mean()),
            'median_length': int(sub['length'].median()),
            'min_length': int(sub['length'].min()),
            'max_length': int(sub['length'].max()),
            'mean_gc': round(sub['gc_content'].mean(), 4),
            'mean_n_pct': round(sub['n_percent'].mean(), 4),
            'duplicates': int(sub['is_duplicate'].sum()),
        })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_class_distribution(df, output_dir):
    """Horizontal bar chart of sequence counts per family."""
    plt, sns = _init_plotting()
    output_dir = Path(output_dir)

    counts = df['family'].value_counts().reindex(FAMILY_ORDER).dropna().astype(int)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(counts.index, counts.values,
                   color=sns.color_palette("colorblind", n_colors=len(counts)))
    for bar, val in zip(bars, counts.values):
        ax.text(val + max(counts.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', fontsize=10)

    ax.set_xlabel('Number of Sequences')
    ax.set_title('Sequence Count per Virus Family')
    ax.invert_yaxis()
    plt.tight_layout()

    path = output_dir / 'class_distribution.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_length_distributions(df, output_dir):
    """Box plot of sequence length per family (log scale)."""
    plt, sns = _init_plotting()
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df, y='family', x='length', hue='family',
        order=FAMILY_ORDER, hue_order=FAMILY_ORDER,
        palette="colorblind", legend=False,
        fliersize=1, ax=ax
    )
    ax.set_xscale('log')
    ax.set_xlabel('Sequence Length (bp, log scale)')
    ax.set_title('Sequence Length Distribution per Family')
    plt.tight_layout()

    path = output_dir / 'length_distributions.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_gc_content(df, output_dir):
    """KDE plot of GC content per family."""
    plt, sns = _init_plotting()
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = sns.color_palette("colorblind", n_colors=len(FAMILY_ORDER))
    for fam, color in zip(FAMILY_ORDER, palette):
        sub = df[df['family'] == fam]
        if sub.empty:
            continue
        sns.kdeplot(sub['gc_content'], label=fam, color=color, ax=ax, fill=True, alpha=0.15)

    ax.set_xlabel('GC Content')
    ax.set_ylabel('Density')
    ax.set_title('GC Content Distribution per Family')
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = output_dir / 'gc_content_distribution.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_quality_metrics(df, output_dir):
    """Strip plot of N-base percentage per family with threshold line."""
    plt, sns = _init_plotting()
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.stripplot(
        data=df, y='family', x='n_percent', hue='family',
        order=FAMILY_ORDER, hue_order=FAMILY_ORDER,
        palette="colorblind", legend=False,
        size=2, alpha=0.4, jitter=True, ax=ax
    )
    ax.axvline(0.05, color='red', linestyle='--', linewidth=1, label='5% threshold')
    ax.set_xlabel('Proportion of Ambiguous Bases (N)')
    ax.set_title('Data Quality: Ambiguous Base Content')
    ax.legend()
    plt.tight_layout()

    path = output_dir / 'quality_metrics.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Preprocessing recommendations
# ---------------------------------------------------------------------------

def generate_preprocessing_recommendations(df):
    """
    Analyze the data and return actionable preprocessing recommendations.
    """
    recs = []

    # --- Class balance ---
    counts = df['family'].value_counts()
    ratio = counts.max() / counts.min()
    if ratio > 5:
        recs.append(
            f"Class imbalance detected (max/min ratio {ratio:.0f}:1). "
            f"Use stratified sampling and class-weighted loss during training."
        )

    # --- Sequence length & chunking ---
    p95 = int(df['length'].quantile(0.95))
    p50 = int(df['length'].median())
    longest_family = df.groupby('family')['length'].median().idxmax()
    longest_median = int(df[df['family'] == longest_family]['length'].median())

    recs.append(
        f"Median sequence length: {p50:,} bp. "
        f"95th percentile: {p95:,} bp. "
        f"Longest family ({longest_family}) has median {longest_median:,} bp."
    )

    if p95 > 6000:
        recs.append(
            "Sequences exceed typical transformer input limits. "
            "Use a sliding-window chunking strategy (e.g. 512-token windows "
            "with 50% overlap) and aggregate chunk predictions per sequence."
        )

    # --- Duplicates ---
    n_dup = df['is_duplicate'].sum()
    if n_dup > 0:
        recs.append(
            f"{n_dup:,} exact duplicate sequences detected. "
            f"Remove before train/test split to prevent data leakage."
        )

    # --- Quality ---
    high_n = (df['n_percent'] > 0.01).sum()
    if high_n > 0:
        recs.append(
            f"{high_n:,} sequences have >1% ambiguous bases. "
            f"Consider stricter filtering or N-masking during tokenization."
        )

    print("\n=== Preprocessing Recommendations ===")
    for i, r in enumerate(recs, 1):
        print(f"  {i}. {r}")

    return recs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def explore_data(input_path, output_dir):
    """
    Run the full exploration pipeline.

    Args:
        input_path: Path to viral_sequences_all.csv
        output_dir: Directory to save figures and summary table
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Viral Sequence Data Exploration ===\n")

    # Load and enrich
    df = load_data(input_path)
    df = compute_sequence_metrics(df)

    # Summary statistics
    summary = print_summary_statistics(df)
    summary_path = output_dir / 'summary_statistics.csv'
    summary.to_csv(summary_path, index=False)
    print(f"\n  Saved summary table to {summary_path}")

    # Plots
    print("\nGenerating plots...")
    plot_class_distribution(df, output_dir)
    plot_length_distributions(df, output_dir)
    plot_gc_content(df, output_dir)
    plot_quality_metrics(df, output_dir)

    # Recommendations
    generate_preprocessing_recommendations(df)

    print(f"\n=== All outputs saved to {output_dir} ===")
    return df, summary


if __name__ == "__main__":
    # Set non-interactive backend before any plotting
    import matplotlib
    matplotlib.use('Agg')

    parser = argparse.ArgumentParser(
        description="Explore viral genomic sequence dataset."
    )
    parser.add_argument(
        '--input', type=str,
        default='data/raw/viral_sequences_all.csv',
        help='Path to combined sequences CSV'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='reports/figures',
        help='Directory to save figures and summaries'
    )
    args = parser.parse_args()

    explore_data(args.input, args.output_dir)
