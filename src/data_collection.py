"""
Download viral genomic sequences from NCBI for virus family classification.
Targets major pathogen families relevant to biosecurity.
"""

import os
from Bio import Entrez, SeqIO
import pandas as pd
from pathlib import Path
import time
from collections import defaultdict
import random

# IMPORTANT: Replace with your email for NCBI
Entrez.email = "samuel.modee@uib.no"  # NCBI requires this

# Virus families we'll classify (biosecurity-relevant)
VIRUS_FAMILIES = {
    'Coronaviridae': 1000,      # SARS, MERS, COVID
    'Orthomyxoviridae': 1000,   # Influenza
    'Filoviridae': 500,         # Ebola, Marburg
    'Flaviviridae': 800,        # Dengue, Zika, Yellow Fever
    'Paramyxoviridae': 800,     # Measles, Nipah
    'Arenaviridae': 500,        # Lassa fever
    'Poxviridae': 500,          # Smallpox, Monkeypox
}

def search_sequences(family_name, max_sequences=1000, min_length=1000, max_length=30000):
    """
    Search NCBI for viral sequences from a specific family.
    
    Args:
        family_name: Virus family name
        max_sequences: Maximum number to retrieve
        min_length: Minimum sequence length (filters out fragments)
        max_length: Maximum sequence length (filters out full genomes that are too long)
    """
    print(f"\nSearching for {family_name} sequences...")
    
    # Search query - targets complete or near-complete sequences
    search_query = (
        f"{family_name}[Organism] AND "
        f"{min_length}:{max_length}[Sequence Length] AND "
        f"complete genome[Title]"
    )
    
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

def collect_data():
    """Main data collection function."""
    
    all_sequences = []
    
    for family, max_count in VIRUS_FAMILIES.items():
        # Search for sequences
        id_list = search_sequences(family, max_sequences=max_count)
        
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
    
    # Combine all data
    df = pd.DataFrame(all_sequences)
    
    # Basic quality filtering
    print("\n=== Data Quality Filtering ===")
    print(f"Total sequences collected: {len(df)}")
    
    # Remove sequences with ambiguous nucleotides (>5%)
    df['n_count'] = df['sequence'].apply(lambda x: x.upper().count('N'))
    df['n_percent'] = df['n_count'] / df['length']
    df_clean = df[df['n_percent'] < 0.05].copy()
    
    print(f"After filtering ambiguous bases: {len(df_clean)}")
    
    # Check class distribution
    print("\n=== Class Distribution ===")
    print(df_clean['family'].value_counts())
    
    # Save to CSV
    output_path = Path('data/raw/viral_sequences_all.csv')
    df_clean.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(df_clean)} sequences to {output_path}")
    
    return df_clean

if __name__ == "__main__":
    # Create directories if needed
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    
    print("=== NCBI Viral Sequence Collection ===")
    print("This will take 20-30 minutes...")
    
    df = collect_data()
    
    print("\n=== Summary Statistics ===")
    print(f"Total sequences: {len(df)}")
    print(f"Number of families: {df['family'].nunique()}")
    print(f"\nSequence length stats:")
    print(df['length'].describe())