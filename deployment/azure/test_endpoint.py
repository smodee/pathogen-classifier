"""
Test client for the deployed pathogen classifier endpoint.

Sends sample DNA sequences to the Azure ML managed endpoint and displays
predictions with confidence scores and latency.

Usage:
    # Using hardcoded test sequences
    python deployment/azure/test_endpoint.py \\
        --endpoint-url https://<endpoint>.inference.ml.azure.com/score \\
        --api-key <key>

    # Using sequences from test CSV
    python deployment/azure/test_endpoint.py \\
        --endpoint-url https://<endpoint>.inference.ml.azure.com/score \\
        --api-key <key> \\
        --csv data/processed/test.csv \\
        --n-samples 10
"""

import argparse
import json
import time
import urllib.request
import urllib.error


# Short representative sequences for quick testing (one per family).
# These are synthetic subsequences — not from the actual test set.
SAMPLE_SEQUENCES = [
    {
        "name": "Poxviridae-like",
        "sequence": "ATGGATCCAACATTTCCATTGGGTTCTACTAAAGAGTTTTGTGAACGAACCGC"
                    "TGATTTCTTCAATCCGGTGGGTAAAGTTGTCGAGGTAGTTTTATCAAGTATCG"
                    "GATGGACTCATCGAGTTAATCTGTCAATGCCTGCTATAGCAGTGTTTGTTAAT"
                    "ACAGGCAGCTCCCTTCGCAATCCAATCCTAACTGCGGATGACTCTGTTTGTAA",
    },
    {
        "name": "Coronaviridae-like",
        "sequence": "ATGTCTGATAATGGACCCCAAAATCAGCGAAATGCACCCCGCATTACGTTTGG"
                    "TGGACCCTCAGATTCAACTGGCAGTAACCAGAATGGAGAACGCAGTGGGGCGC"
                    "GATCAAAACAACGTCGGCCCCAAGGTTTACCCAATAATACTGCGTCTTGGTTCA"
                    "CCGCTCTCACTCAACATGGCAAGGAAGACCTTAAATTCCCTCGAGGACAAGGCG",
    },
    {
        "name": "Flaviviridae-like",
        "sequence": "ATGAAAAACCCCAAAGAAGAAATCCGGAGGATTCCGGATTGTCAATATGCTAAA"
                    "ACGCGGAGTAGCCCGTAAGTCACCCAGCCTACTTTCAATCAGGAGGCTCTCAA"
                    "TGACCTGAGAATACCATTTTCCAACAAGCTGATGCAGTGTTATTCCAGTCTTT"
                    "GCCATCTCTTAAGGCGATTACAAATATCTTTGATCAAAGCCTTGATTTCACATT",
    },
]


def test_endpoint(endpoint_url, api_key, sequences, names=None):
    """Send sequences to the endpoint and display results.

    Parameters
    ----------
    endpoint_url : str
        Scoring URI from deploy.py output.
    api_key : str
        API key from deploy.py output.
    sequences : list of str
        DNA sequences to classify.
    names : list of str, optional
        Display names for each sequence.
    """
    if names is None:
        names = [f"seq_{i}" for i in range(len(sequences))]

    payload = json.dumps({"sequences": sequences}).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    print(f"Sending {len(sequences)} sequences to {endpoint_url}")
    print(f"Payload size: {len(payload):,} bytes\n")

    req = urllib.request.Request(endpoint_url, data=payload, headers=headers)

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            status = response.status
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP Error {e.code}: {e.reason}")
        print(f"Response body: {body}")
        return
    except urllib.error.URLError as e:
        print(f"Connection error: {e.reason}")
        print("Check that the endpoint URL is correct and the endpoint is running.")
        return

    elapsed = time.time() - start

    # Azure ML may return a JSON string (score.py returns json.dumps())
    # so we may need to parse it again
    if isinstance(result, str):
        result = json.loads(result)

    if "error" in result:
        print(f"Error from endpoint: {result['error']}")
        return

    predictions = result.get("predictions", [])

    # Display results
    print(f"{'Name':<25} {'Predicted Family':<20} {'Confidence':>10} {'Latency':>10}")
    print("-" * 70)

    for i, pred in enumerate(predictions):
        name = names[i] if i < len(names) else f"seq_{i}"
        family = pred["predicted_family"]
        conf = pred["confidence"]
        print(f"{name:<25} {family:<20} {conf:>9.4f} ", end="")
        if i == 0:
            print(f"{elapsed:>9.3f}s")
        else:
            print()

    print(f"\nTotal latency: {elapsed:.3f}s ({len(sequences)} sequences)")
    print(f"HTTP status: {status}")

    # Show probability details for first prediction
    if predictions:
        print(f"\nProbability breakdown (first sequence):")
        probs = predictions[0]["probabilities"]
        for family, prob in sorted(probs.items(), key=lambda x: -x[1]):
            bar = "#" * int(prob * 50)
            print(f"  {family:<20} {prob:.4f} {bar}")


def load_csv_samples(csv_path, n_samples=5):
    """Load sample sequences from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to CSV with 'sequence' and 'family' columns.
    n_samples : int
        Number of samples to load.

    Returns
    -------
    tuple of (list of str, list of str)
        (sequences, names) where names include the true family.
    """
    # Minimal CSV parsing — no pandas dependency required
    sequences = []
    names = []

    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        seq_idx = header.index("sequence")
        fam_idx = header.index("family")

        for line_num, line in enumerate(f):
            if line_num >= n_samples:
                break
            parts = line.strip().split(",")
            sequences.append(parts[seq_idx])
            names.append(f"{parts[fam_idx]}_{line_num}")

    return sequences, names


def main():
    parser = argparse.ArgumentParser(
        description="Test the deployed pathogen classifier endpoint",
    )
    parser.add_argument(
        "--endpoint-url", required=True,
        help="Scoring URI from deploy.py output",
    )
    parser.add_argument(
        "--api-key", required=True,
        help="API key from deploy.py output",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Optional: path to test CSV for real sequences",
    )
    parser.add_argument(
        "--n-samples", type=int, default=5,
        help="Number of samples to load from CSV (default: 5)",
    )

    args = parser.parse_args()

    if args.csv:
        print(f"Loading {args.n_samples} samples from {args.csv}...")
        sequences, names = load_csv_samples(args.csv, args.n_samples)
    else:
        print("Using built-in test sequences...\n")
        sequences = [s["sequence"] for s in SAMPLE_SEQUENCES]
        names = [s["name"] for s in SAMPLE_SEQUENCES]

    test_endpoint(args.endpoint_url, args.api_key, sequences, names)


if __name__ == "__main__":
    main()
