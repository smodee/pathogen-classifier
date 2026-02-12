"""
Download viral genomic sequences from NCBI for virus family classification.
Targets major pathogen families relevant to biosecurity.
"""

import os
import argparse
from Bio import Entrez, SeqIO
import pandas as pd
from pathlib import Path
import time
from collections import defaultdict
import random

# NCBI requires an email for API access
Entrez.email = os.environ.get("NCBI_EMAIL")
if not Entrez.email:
    raise SystemExit("Error: Set NCBI_EMAIL environment variable before running.\n"
                     "  Windows:  set NCBI_EMAIL=your.email@example.com\n"
                     "  Linux/Mac: export NCBI_EMAIL=your.email@example.com")

# Per-family search config: max_seq, min/max length, and optional title filter.
# Tuned for each family's genome characteristics.
VIRUS_FAMILIES = {
    'Coronaviridae':    {'max_seq': 1000, 'min_len': 1000,  'max_len': 32000,  'query': 'complete genome'},  # SARS, MERS, COVID (~30kb)
    'Orthomyxoviridae': {'max_seq': 1000, 'min_len': 500,   'max_len': 25000,  'query': None},               # Influenza (8 segments, ~0.9-2.3kb each)
    'Filoviridae':      {'max_seq': 500,  'min_len': 1000,  'max_len': 30000,  'query': 'complete genome'},  # Ebola, Marburg (~19kb)
    'Flaviviridae':     {'max_seq': 800,  'min_len': 1000,  'max_len': 30000,  'query': 'complete genome'},  # Dengue, Zika (~11kb)
    'Paramyxoviridae':  {'max_seq': 800,  'min_len': 1000,  'max_len': 30000,  'query': 'complete genome'},  # Measles, Nipah (~15-16kb)
    'Arenaviridae':     {'max_seq': 500,  'min_len': 500,   'max_len': 15000,  'query': None},               # Lassa fever (2 segments, ~3.5-7kb)
    'Poxviridae':       {'max_seq': 500,  'min_len': 50000, 'max_len': 250000, 'query': 'complete genome'},  # Smallpox, Monkeypox (~130-200kb)
}

def search_sequences(family_name, max_sequences=1000, min_length=1000,
                     max_length=30000, title_filter="complete genome"):
    """
    Search NCBI for viral sequences from a specific family.

    Args:
        family_name: Virus family name
        max_sequences: Maximum number to retrieve
        min_length: Minimum sequence length (filters out fragments)
        max_length: Maximum sequence length
        title_filter: Optional title search term (e.g. "complete genome").
                      Set to None to omit title filtering.
    """
    print(f"\nSearching for {family_name} sequences...")

    # Build search query — title filter is optional per family
    search_query = (
        f"{family_name}[Organism] AND "
        f"{min_length}:{max_length}[Sequence Length]"
    )
    if title_filter:
        search_query += f" AND {title_filter}[Title]"
    
    try:
        # Search NCBI
        handle = Entrez.esearch(
            db="nucleotide",
            term=search_query,
            retmax=max_sequences,
            sort="relevance"
        )
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record["IdList"]
        print(f"Found {len(id_list)} sequences for {family_name}")
        
        return id_list
    
    except Exception as e:
        print(f"Error searching {family_name}: {e}")
        return []

def fetch_sequences(id_list, family_name, batch_size=100):
    """
    Fetch actual sequence data from NCBI.
    Downloads in batches to avoid timeouts.
    """
    sequences = []
    
    # Download in batches
    for i in range(0, len(id_list), batch_size):
        batch_ids = id_list[i:i+batch_size]
        print(f"  Fetching batch {i//batch_size + 1} ({len(batch_ids)} sequences)...")
        
        try:
            # Fetch sequences
            handle = Entrez.efetch(
                db="nucleotide",
                id=batch_ids,
                rettype="fasta",
                retmode="text"
            )
            
            # Parse FASTA
            batch_records = list(SeqIO.parse(handle, "fasta"))
            handle.close()
            
            # Store with metadata
            for record in batch_records:
                sequences.append({
                    'id': record.id,
                    'description': record.description,
                    'sequence': str(record.seq),
                    'length': len(record.seq),
                    'family': family_name
                })
            
            # Be nice to NCBI servers
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  Error fetching batch: {e}")
            time.sleep(2)  # Wait longer on error
            continue
    
    return sequences

def collect_data(families=None):
    """
    Main data collection function.

    Args:
        families: List of family names to collect. Defaults to all families.
    """
    if families is None:
        families = list(VIRUS_FAMILIES.keys())

    all_sequences = []

    for family in families:
        if family not in VIRUS_FAMILIES:
            print(f"Warning: Unknown family '{family}', skipping")
            continue

        cfg = VIRUS_FAMILIES[family]

        # Search for sequences with per-family parameters
        id_list = search_sequences(
            family,
            max_sequences=cfg['max_seq'],
            min_length=cfg['min_len'],
            max_length=cfg['max_len'],
            title_filter=cfg['query'],
        )

        if not id_list:
            print(f"Warning: No sequences found for {family}")
            continue

        # Fetch sequences
        sequences = fetch_sequences(id_list, family)
        all_sequences.extend(sequences)

        print(f"Collected {len(sequences)} sequences for {family}")

        # Save progress incrementally (in case of crashes)
        df_progress = pd.DataFrame(sequences)
        df_progress.to_csv(f'data/raw/{family}_sequences.csv', index=False)

    # Rebuild combined file from ALL per-family CSVs (not just this run)
    print("\n=== Rebuilding combined dataset ===")
    all_csvs = list(Path('data/raw').glob('*_sequences.csv'))
    df = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)

    # Basic quality filtering
    print(f"Total sequences across all families: {len(df)}")

    # Remove sequences with ambiguous nucleotides (>5%)
    df['n_count'] = df['sequence'].apply(lambda x: x.upper().count('N'))
    df['n_percent'] = df['n_count'] / df['length']
    df_clean = df[df['n_percent'] < 0.05].copy()

    print(f"After filtering ambiguous bases: {len(df_clean)}")

    # Check class distribution
    print("\n=== Class Distribution ===")
    print(df_clean['family'].value_counts())

    # Save combined file
    output_path = Path('data/raw/viral_sequences_all.csv')
    df_clean.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(df_clean)} sequences to {output_path}")

    return df_clean

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download viral genomic sequences from NCBI."
    )
    parser.add_argument(
        '--families', nargs='+', default=None,
        choices=list(VIRUS_FAMILIES.keys()),
        help='Specific families to collect (default: all). '
             'Useful for re-collecting families with updated parameters.'
    )
    args = parser.parse_args()

    # Create directories if needed
    Path('data/raw').mkdir(parents=True, exist_ok=True)

    target = args.families if args.families else list(VIRUS_FAMILIES.keys())
    print("=== NCBI Viral Sequence Collection ===")
    print(f"Collecting: {', '.join(target)}")
    print("This may take 20-30 minutes for a full run...\n")

    df = collect_data(families=args.families)

    print("\n=== Summary Statistics ===")
    print(f"Total sequences: {len(df)}")
    print(f"Number of families: {df['family'].nunique()}")
    print(f"\nSequence length stats:")
    print(df['length'].describe())