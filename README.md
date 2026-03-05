# Viral Pathogen Classification using Genomic Sequences

An end-to-end machine learning pipeline for classifying viral pathogens into 7 biosecurity-relevant virus families from raw genomic sequences. Compares a classical k-mer frequency baseline against a fine-tuned Nucleotide Transformer (NT v2 50M).

Built as a portfolio project for a Data Scientist position at the European Centre for Disease Prevention and Control (ECDC).

## Architecture

```
NCBI Nucleotide DB
        |
        v
  [Data Acquisition]  ------>  4,791 raw viral sequences
        |
  [Preprocessing]     ------>  Dedup, 5 kb chunking, stratified split
        |
        v
  43,881 chunks (train/val/test)
       / \
      /   \
     v     v
  [k-mer + XGBoost]    [NT v2 50M Transformer]
  5-mer frequencies     Frozen encoder + head
  1,024 features        266K trainable params
  CPU training          Kaggle T4 GPU
     |                    |
     v                    v
  [Unified Evaluation]  ------>  Metrics, confusion matrices, comparison
        |
        v
  [Azure ML Deployment] ------>  REST API (XGBoost baseline)
  Standard_DS2_v2 CPU            POST /score {"sequences": [...]}
```

## Results

Evaluated on a held-out test set (6,553 sequences):

| Model | Macro F1 | Balanced Accuracy | Accuracy | Trainable Params |
|---|---|---|---|---|
| **k-mer + XGBoost (baseline)** | **0.9979** | **0.9976** | **0.9994** | 1,024 features |
| NT v2 50M (frozen encoder) | 0.9640 | 0.9628 | 0.9881 | 266K / 53.8M |

### Per-class F1 scores (test set)

| Family | Baseline | NT v2 | Test Samples |
|---|---|---|---|
| Poxviridae | 1.000 | 0.999 | 3,653 |
| Coronaviridae | 1.000 | 0.999 | 1,199 |
| Paramyxoviridae | 0.999 | 0.960 | 699 |
| Filoviridae | 1.000 | 0.933 | 371 |
| Flaviviridae | 0.999 | 0.976 | 443 |
| Arenaviridae | 0.989 | 0.890 | 87 |
| Orthomyxoviridae | 1.000 | 0.990 | 101 |

### Key findings

- The k-mer frequency baseline achieves near-perfect classification, indicating strong compositional signatures across virus families.
- The NT v2 transformer with frozen encoder (only 0.5% of parameters trainable) performs well but does not surpass the baseline. Unfreezing encoder layers or using a larger model may close the gap.
- Both models handle class imbalance effectively via inverse-frequency weighting (Arenaviridae comprises only 1.3% of data).

## Project Structure

```
pathogen-classifier/
  data/
    raw/                  # NCBI viral sequences (gitignored)
    processed/            # Train/val/test splits (gitignored)
  src/
    data_exploration.py   # EDA, statistics, visualizations
    data_preprocessing.py # Dedup, chunking, stratified splitting
    baseline_model.py     # k-mer frequency + XGBoost
    dataset.py            # PyTorch Dataset + NT v2 tokenization
    augmentation.py       # Genomic data augmentation (5 transforms)
    train_transformer.py  # NT v2 50M fine-tuning with HuggingFace Trainer
    evaluate.py           # Unified evaluation, metrics, comparison
    export_model.py       # ONNX export + Azure ML packaging
  deployment/
    azure/
      deploy.py           # Azure ML deployment orchestration
      score.py            # XGBoost scoring script for managed endpoint
      test_endpoint.py    # Client script to test the live endpoint
      conda_baseline.yml  # Lightweight conda environment (no torch)
  configs/
    train_nt.json         # Transformer training hyperparameters
  models/                 # Trained models (gitignored)
  reports/                # Evaluation outputs (gitignored)
  notebooks/              # Kaggle training & verification scripts
  tests/                  # Unit and integration tests
```

## Data

- **Source:** NCBI Nucleotide database via BioPython Entrez API
- **7 virus families:** Poxviridae, Coronaviridae, Paramyxoviridae, Filoviridae, Flaviviridae, Arenaviridae, Orthomyxoviridae
- **Pipeline:** 4,791 raw sequences -> deduplication -> 5,000 bp chunking -> stratified split
- **Splits:** 30,774 train / 6,554 val / 6,553 test chunks

## Models

### Baseline: k-mer + XGBoost

Extracts normalized 5-mer frequency vectors (1,024 features) and trains a gradient-boosted classifier with inverse-frequency class weights.

```bash
python -m src.baseline_model --train data/processed/train.csv --val data/processed/val.csv
```

### Transformer: Nucleotide Transformer v2 50M

Fine-tunes [InstaDeepAI/nucleotide-transformer-v2-50m-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species) with a frozen encoder (only classification head trainable). Uses class-weighted cross-entropy loss and genomic data augmentation (reverse complement, random crop, sequencing noise, N-masking).

Trained on Kaggle free T4 GPU. See `notebooks/kaggle_nt_v2_training.py`.

### Evaluation

```bash
# Evaluate a single model
python -m src.evaluate --model-type baseline --model-dir models/baseline --test data/processed/test.csv

# Compare models
python -m src.evaluate --compare reports/baseline_metrics.json reports/transformer_metrics.json
```

### ONNX Export

```bash
python -m src.export_model --model-dir models/nt-v2 --format onnx --output models/exports/nt_v2.onnx --validate --test-csv data/processed/test.csv
```

## Deployment

The best-performing model (k-mer + XGBoost) is deployed as a REST API on Azure ML. The baseline was chosen over the transformer for deployment because it achieves higher accuracy (99.79% vs 96.40% macro F1), runs on CPU-only infrastructure, and has a 2.2 MB model footprint requiring no PyTorch dependency.

### Prerequisites

1. [Azure account](https://azure.microsoft.com/free/) (free tier or Pay-As-You-Go)
2. Azure CLI: `winget install Microsoft.AzureCLI`
3. Authenticate and set up resources:

```bash
az login
az provider register --namespace Microsoft.MachineLearningServices
az provider register --namespace Microsoft.PolicyInsights
az group create --name pathogen-rg --location westeurope
az ml workspace create --name pathogen-ws --resource-group pathogen-rg
```

### Deploy

```bash
python deployment/azure/deploy.py \
    --subscription-id <your-subscription-id> \
    --resource-group pathogen-rg \
    --workspace-name pathogen-ws
```

### Test the endpoint

```bash
python deployment/azure/test_endpoint.py \
    --endpoint-url <scoring-uri> \
    --api-key <api-key>
```

#### API request/response example

```bash
curl -X POST <scoring-uri> \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{"sequences": ["ATGGATCCAACATTTCCATTGGGTTCTACT..."]}'
```

```json
{
  "predictions": [
    {
      "predicted_family": "Poxviridae",
      "confidence": 0.9988,
      "probabilities": {
        "Poxviridae": 0.9988,
        "Coronaviridae": 0.0003,
        "Paramyxoviridae": 0.0002,
        "Filoviridae": 0.0001,
        "Flaviviridae": 0.0002,
        "Arenaviridae": 0.0001,
        "Orthomyxoviridae": 0.0003
      }
    }
  ]
}
```

### Teardown (stop billing)

```bash
python deployment/azure/deploy.py --teardown \
    --subscription-id <your-subscription-id> \
    --resource-group pathogen-rg \
    --workspace-name pathogen-ws
```

Estimated cost: ~$1-2 for a demo session (Standard_DS2_v2 at $0.146/hr).

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Testing

107 tests covering data augmentation, baseline model, evaluation metrics, and scoring script validation.

```bash
pytest tests/ -v
```

## Tech Stack

- Python 3.13, PyTorch 2.10, HuggingFace Transformers 4.x
- XGBoost, scikit-learn, pandas, NumPy
- ONNX / ONNX Runtime for model export
- BioPython for NCBI data collection
- Azure ML managed endpoints for REST API deployment

## License

MIT
