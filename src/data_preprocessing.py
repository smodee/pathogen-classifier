"""
Preprocess raw viral sequences into model-ready train/val/test splits.

Pipeline: Deduplicate -> Quality filter -> Chunk -> Stratified split -> Save

Usage:
    python src/data_preprocessing.py
    python src/data_preprocessing.py --input data/raw/viral_sequences_all.csv
    python src/data_preprocessing.py --chunk-size 3000 --overlap 0.5 --split-ratio 0.7 0.15 0.15
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from data_exploration import FAMILY_ORDER, load_data, compute_sequence_metrics


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(df, log):
    """
    Remove exact duplicate sequences.

    Args:
        df: DataFrame with is_duplicate column from compute_sequence_metrics.
        log: List to append log messages to.

    Returns:
        Deduplicated DataFrame.
    """
    msg = "=== Step 1: Deduplication ==="
    print(msg)
    log.append(msg)

    n_before = len(df)
    removed_per_family = (
        df[df['is_duplicate']]
        .groupby('family')
        .size()
        .reindex(FAMILY_ORDER, fill_value=0)
    )

    for fam in FAMILY_ORDER:
        n = removed_per_family.get(fam, 0)
        if n > 0:
            total = len(df[df['family'] == fam])
            line = f"  {fam}: removed {n} duplicates ({n}/{total}, {n/total*100:.1f}%)"
            print(line)
            log.append(line)

    df = df[~df['is_duplicate']].copy()
    n_after = len(df)

    summary = f"  Total: {n_before} -> {n_after} (removed {n_before - n_after})"
    print(summary)
    log.append(summary)

    return df


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------

def quality_filter(df, max_n_percent, log):
    """
    Remove sequences with too many ambiguous bases.

    Args:
        df: DataFrame with n_percent column.
        max_n_percent: Maximum allowed proportion of N bases (exclusive).
        log: List to append log messages to.

    Returns:
        Filtered DataFrame.
    """
    msg = f"\n=== Step 2: Quality Filtering (N-base threshold: {max_n_percent*100:.0f}%) ==="
    print(msg)
    log.append(msg)

    n_before = len(df)
    mask = df['n_percent'] > max_n_percent

    removed_per_family = (
        df[mask]
        .groupby('family')
        .size()
        .reindex(FAMILY_ORDER, fill_value=0)
    )

    for fam in FAMILY_ORDER:
        n = removed_per_family.get(fam, 0)
        if n > 0:
            total = len(df[df['family'] == fam])
            line = f"  {fam}: removed {n} ({n/total*100:.1f}%)"
            print(line)
            log.append(line)

    df = df[~mask].copy()
    n_after = len(df)

    summary = f"  Total: {n_before} -> {n_after} (removed {n_before - n_after})"
    print(summary)
    log.append(summary)

    return df


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_sequences(df, chunk_size, overlap, log):
    """
    Split sequences into fixed-size windows with overlap.

    Sequences shorter than chunk_size are kept as a single chunk.
    Each chunk inherits the family label from its parent sequence.

    Args:
        df: DataFrame with id, sequence, length, family columns.
        chunk_size: Window size in base pairs.
        overlap: Fraction of overlap between consecutive windows (0–1).
        log: List to append log messages to.

    Returns:
        DataFrame of chunks with columns:
            id, parent_sequence_id, chunk_index, sequence, length, family
    """
    stride = int(chunk_size * (1 - overlap))
    msg = (
        f"\n=== Step 3: Chunking (size={chunk_size} bp, "
        f"overlap={overlap*100:.0f}%, stride={stride} bp) ==="
    )
    print(msg)
    log.append(msg)

    chunks = []
    for _, row in df.iterrows():
        seq = row['sequence']
        seq_len = len(seq)
        parent_id = row['id']
        family = row['family']

        if seq_len <= chunk_size:
            # Keep short sequences as a single chunk
            chunks.append({
                'id': f"{parent_id}_chunk0",
                'parent_sequence_id': parent_id,
                'chunk_index': 0,
                'sequence': seq,
                'length': len(seq),
                'family': family,
            })
        else:
            # Sliding window
            chunk_idx = 0
            for start in range(0, seq_len - chunk_size + 1, stride):
                chunk_seq = seq[start:start + chunk_size]
                chunks.append({
                    'id': f"{parent_id}_chunk{chunk_idx}",
                    'parent_sequence_id': parent_id,
                    'chunk_index': chunk_idx,
                    'sequence': chunk_seq,
                    'length': len(chunk_seq),
                    'family': family,
                })
                chunk_idx += 1

            # If the last window didn't reach the end, add a final chunk
            # aligned to the sequence end to avoid losing trailing bases
            last_start = start + chunk_size
            if last_start < seq_len:
                final_start = seq_len - chunk_size
                chunk_seq = seq[final_start:]
                chunks.append({
                    'id': f"{parent_id}_chunk{chunk_idx}",
                    'parent_sequence_id': parent_id,
                    'chunk_index': chunk_idx,
                    'sequence': chunk_seq,
                    'length': len(chunk_seq),
                    'family': family,
                })

    df_chunks = pd.DataFrame(chunks)

    # Per-family stats
    n_parents = df['family'].value_counts().reindex(FAMILY_ORDER, fill_value=0)
    n_chunks = df_chunks['family'].value_counts().reindex(FAMILY_ORDER, fill_value=0)
    for fam in FAMILY_ORDER:
        np_ = n_parents.get(fam, 0)
        nc = n_chunks.get(fam, 0)
        if np_ > 0:
            line = f"  {fam}: {np_} sequences -> {nc} chunks ({nc/np_:.1f}x)"
            print(line)
            log.append(line)

    summary = f"  Total: {len(df)} sequences -> {len(df_chunks)} chunks"
    print(summary)
    log.append(summary)

    return df_chunks


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(df_chunks, split_ratio, seed, log):
    """
    Split chunks into train/val/test sets at the parent-sequence level.

    All chunks from the same parent sequence go into the same split,
    preventing data leakage.

    Args:
        df_chunks: DataFrame with parent_sequence_id and family columns.
        split_ratio: Tuple of (train, val, test) fractions summing to 1.
        seed: Random seed for reproducibility.
        log: List to append log messages to.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    train_frac, val_frac, test_frac = split_ratio
    msg = (
        f"\n=== Step 4: Stratified Split "
        f"(train={train_frac}, val={val_frac}, test={test_frac}, seed={seed}) ==="
    )
    print(msg)
    log.append(msg)

    # Build a parent-level table for stratified splitting
    parents = (
        df_chunks
        .groupby('parent_sequence_id')['family']
        .first()
        .reset_index()
    )

    # First split: train vs (val+test)
    val_test_frac = val_frac + test_frac
    train_parents, valtest_parents = train_test_split(
        parents,
        test_size=val_test_frac,
        stratify=parents['family'],
        random_state=seed,
    )

    # Second split: val vs test
    relative_test_frac = test_frac / val_test_frac
    val_parents, test_parents = train_test_split(
        valtest_parents,
        test_size=relative_test_frac,
        stratify=valtest_parents['family'],
        random_state=seed,
    )

    # Map parent IDs back to chunks
    train_ids = set(train_parents['parent_sequence_id'])
    val_ids = set(val_parents['parent_sequence_id'])
    test_ids = set(test_parents['parent_sequence_id'])

    train_df = df_chunks[df_chunks['parent_sequence_id'].isin(train_ids)].copy()
    val_df = df_chunks[df_chunks['parent_sequence_id'].isin(val_ids)].copy()
    test_df = df_chunks[df_chunks['parent_sequence_id'].isin(test_ids)].copy()

    # Log split statistics
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        n_parents = split_df['parent_sequence_id'].nunique()
        n_chunks = len(split_df)
        line = f"  {name}: {n_parents} sequences, {n_chunks} chunks"
        print(line)
        log.append(line)

        # Per-family breakdown
        fam_counts = split_df['family'].value_counts().reindex(FAMILY_ORDER, fill_value=0)
        for fam in FAMILY_ORDER:
            c = fam_counts.get(fam, 0)
            if c > 0:
                detail = f"    {fam}: {c} chunks"
                print(detail)
                log.append(detail)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess(input_path, output_dir, chunk_size, overlap, split_ratio, seed):
    """
    Run the full preprocessing pipeline.

    Args:
        input_path: Path to viral_sequences_all.csv.
        output_dir: Directory to save processed outputs.
        chunk_size: Chunk window size in base pairs.
        overlap: Fraction of overlap between windows.
        split_ratio: Tuple of (train, val, test) fractions.
        seed: Random seed for reproducibility.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log = []
    header = "=== Viral Sequence Preprocessing Pipeline ==="
    print(header)
    log.append(header)

    params = (
        f"Parameters: chunk_size={chunk_size}, overlap={overlap}, "
        f"split_ratio={split_ratio}, seed={seed}"
    )
    print(params)
    log.append(params)

    # Load and compute metrics
    df = load_data(input_path)
    df = compute_sequence_metrics(df)

    log.append(f"\nInput: {len(df)} sequences, {df['family'].nunique()} families")

    # Step 1: Deduplication
    df = deduplicate(df, log)

    # Step 2: Quality filtering
    df = quality_filter(df, max_n_percent=0.02, log=log)

    # Step 3: Chunking
    df_chunks = chunk_sequences(df, chunk_size, overlap, log)

    # Step 4: Stratified split
    train_df, val_df, test_df = stratified_split(df_chunks, split_ratio, seed, log)

    # Step 5: Save outputs
    msg = "\n=== Step 5: Saving Outputs ==="
    print(msg)
    log.append(msg)

    output_cols = ['id', 'parent_sequence_id', 'chunk_index', 'sequence', 'length', 'family']

    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        path = output_dir / f'{name}.csv'
        split_df[output_cols].to_csv(path, index=False)
        line = f"  Saved {path} ({len(split_df)} chunks)"
        print(line)
        log.append(line)

    # Save preprocessing report
    report_path = output_dir / 'preprocessing_report.txt'
    report_path.write_text('\n'.join(log), encoding='utf-8')
    print(f"  Saved {report_path}")

    print("\n=== Preprocessing Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess viral sequences: deduplicate, chunk, and split."
    )
    parser.add_argument(
        '--input', type=str,
        default='data/raw/viral_sequences_all.csv',
        help='Path to combined sequences CSV'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='data/processed',
        help='Directory to save processed outputs'
    )
    parser.add_argument(
        '--chunk-size', type=int,
        default=5000,
        help='Chunk window size in base pairs (default: 5000)'
    )
    parser.add_argument(
        '--overlap', type=float,
        default=0.5,
        help='Fraction of overlap between chunks (default: 0.5)'
    )
    parser.add_argument(
        '--split-ratio', type=float, nargs=3,
        default=[0.7, 0.15, 0.15],
        metavar=('TRAIN', 'VAL', 'TEST'),
        help='Train/val/test split ratios (default: 0.7 0.15 0.15)'
    )
    parser.add_argument(
        '--seed', type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    args = parser.parse_args()

    # Validate split ratio
    ratio_sum = sum(args.split_ratio)
    if abs(ratio_sum - 1.0) > 1e-6:
        parser.error(f"Split ratios must sum to 1.0, got {ratio_sum}")

    preprocess(
        input_path=args.input,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        split_ratio=tuple(args.split_ratio),
        seed=args.seed,
    )
