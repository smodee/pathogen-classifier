"""
Azure ML deployment script for the pathogen classifier (XGBoost baseline).

Deploys the trained k-mer + XGBoost model as a managed online endpoint
using the Azure AI ML SDK v2.

Usage:
    # Deploy
    python deployment/azure/deploy.py \\
        --subscription-id <sub> \\
        --resource-group pathogen-rg \\
        --workspace-name pathogen-ws

    # Teardown (stop billing!)
    python deployment/azure/deploy.py \\
        --teardown \\
        --subscription-id <sub> \\
        --resource-group pathogen-rg \\
        --workspace-name pathogen-ws

Prerequisites:
    1. Azure free account (azure.microsoft.com)
    2. Azure CLI installed: winget install Microsoft.AzureCLI
    3. Authenticated: az login
    4. pip install azure-ai-ml azure-identity
"""

import argparse
import sys
import time
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)
from azure.identity import DefaultAzureCredential


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
DEPLOY_DIR = Path(__file__).resolve().parent

ENDPOINT_NAME = "pathogen-classifier"
DEPLOYMENT_NAME = "baseline-v1"
MODEL_NAME = "pathogen-baseline-xgboost"
INSTANCE_TYPE = "Standard_DS2_v2"
INSTANCE_COUNT = 1


# ============================================================================
# Azure ML helpers
# ============================================================================


def get_ml_client(subscription_id, resource_group, workspace_name):
    """Authenticate and return an MLClient.

    Uses DefaultAzureCredential which supports:
    - Azure CLI (az login)
    - Environment variables
    - Managed identity
    - Visual Studio Code

    Parameters
    ----------
    subscription_id : str
        Azure subscription ID.
    resource_group : str
        Azure resource group name.
    workspace_name : str
        Azure ML workspace name.

    Returns
    -------
    MLClient
        Authenticated ML client.
    """
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
    print(f"Connected to workspace: {workspace_name}")
    return ml_client


def register_model(ml_client, model_path=None):
    """Register the XGBoost model in Azure ML.

    Parameters
    ----------
    ml_client : MLClient
        Authenticated ML client.
    model_path : Path, optional
        Path to baseline model directory. Defaults to models/baseline/.

    Returns
    -------
    Model
        Registered Azure ML model.
    """
    if model_path is None:
        model_path = MODEL_DIR

    # Verify required files exist
    required_files = ["xgboost_model.json", "label_encoder.json", "kmer_params.json"]
    for fname in required_files:
        fpath = model_path / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Missing {fname} in {model_path}. "
                f"Train the baseline model first: "
                f"python -m src.baseline_model --train data/processed/train.csv "
                f"--val data/processed/val.csv"
            )

    model = Model(
        name=MODEL_NAME,
        path=str(model_path),
        description=(
            "XGBoost baseline classifier for viral pathogen identification. "
            "Uses 5-mer frequency features to classify genomic sequences "
            "into 7 virus families (macro F1: 0.9979 on test set)."
        ),
    )

    registered = ml_client.models.create_or_update(model)
    print(f"Model registered: {registered.name} (version {registered.version})")
    return registered


def deploy_endpoint(ml_client, model):
    """Create a managed online endpoint and deploy the model.

    Parameters
    ----------
    ml_client : MLClient
        Authenticated ML client.
    model : Model
        Registered Azure ML model.

    Returns
    -------
    tuple of (str, str)
        (scoring_uri, api_key) for testing the endpoint.
    """
    # Create endpoint
    print(f"\nCreating endpoint: {ENDPOINT_NAME}...")
    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="Viral pathogen classifier — classifies DNA sequences into 7 virus families",
        auth_mode="key",
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Endpoint created: {ENDPOINT_NAME}")

    # Create environment
    env = Environment(
        name="pathogen-baseline-env",
        conda_file=str(DEPLOY_DIR / "conda_baseline.yml"),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        description="Lightweight environment for XGBoost baseline scoring",
    )

    # Create deployment
    print(f"\nCreating deployment: {DEPLOYMENT_NAME}...")
    print(f"  Instance type: {INSTANCE_TYPE}")
    print(f"  Instance count: {INSTANCE_COUNT}")
    print(f"  This may take 5-10 minutes...")

    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=model,
        environment=env,
        code_configuration=CodeConfiguration(
            code=str(DEPLOY_DIR),
            scoring_script="score.py",
        ),
        instance_type=INSTANCE_TYPE,
        instance_count=INSTANCE_COUNT,
    )

    start = time.time()
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    elapsed = time.time() - start
    print(f"Deployment created in {elapsed:.0f}s")

    # Route 100% traffic to this deployment
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print("Traffic routing: 100% -> baseline-v1")

    # Get scoring URI and key
    endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    scoring_uri = endpoint.scoring_uri

    keys = ml_client.online_endpoints.get_keys(ENDPOINT_NAME)
    api_key = keys.primary_key

    print(f"\n{'=' * 60}")
    print("DEPLOYMENT SUCCESSFUL")
    print(f"{'=' * 60}")
    print(f"  Scoring URI: {scoring_uri}")
    print(f"  API Key:     {api_key}")
    print(f"\nTest with:")
    print(f"  python deployment/azure/test_endpoint.py \\")
    print(f"    --endpoint-url {scoring_uri} \\")
    print(f"    --api-key {api_key}")
    print(f"\nIMPORTANT: When done, teardown to stop billing:")
    print(f"  python deployment/azure/deploy.py --teardown \\")
    print(f"    --subscription-id <sub> --resource-group <rg> --workspace-name <ws>")

    return scoring_uri, api_key


def teardown(ml_client):
    """Delete the endpoint and deployment to stop billing.

    Parameters
    ----------
    ml_client : MLClient
        Authenticated ML client.
    """
    print(f"\nDeleting endpoint: {ENDPOINT_NAME}...")
    print("  This will delete the endpoint and all its deployments.")

    try:
        ml_client.online_endpoints.begin_delete(name=ENDPOINT_NAME).result()
        print(f"Endpoint '{ENDPOINT_NAME}' deleted. Billing stopped.")
    except Exception as e:
        if "not found" in str(e).lower():
            print(f"Endpoint '{ENDPOINT_NAME}' not found (already deleted?).")
        else:
            raise


def get_logs(ml_client):
    """Retrieve deployment logs for debugging.

    Parameters
    ----------
    ml_client : MLClient
        Authenticated ML client.
    """
    print(f"\nFetching logs for {DEPLOYMENT_NAME}...")
    try:
        logs = ml_client.online_deployments.get_logs(
            name=DEPLOYMENT_NAME,
            endpoint_name=ENDPOINT_NAME,
            lines=100,
        )
        print(logs)
    except Exception as e:
        print(f"Error fetching logs: {e}")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Deploy pathogen classifier to Azure ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Deploy\n"
            "  python deployment/azure/deploy.py \\\n"
            "    --subscription-id abc-123 --resource-group pathogen-rg\n\n"
            "  # Teardown\n"
            "  python deployment/azure/deploy.py --teardown \\\n"
            "    --subscription-id abc-123 --resource-group pathogen-rg\n"
        ),
    )

    parser.add_argument(
        "--subscription-id", required=True,
        help="Azure subscription ID (from 'az account show')",
    )
    parser.add_argument(
        "--resource-group", default="pathogen-rg",
        help="Azure resource group name (default: pathogen-rg)",
    )
    parser.add_argument(
        "--workspace-name", default="pathogen-ws",
        help="Azure ML workspace name (default: pathogen-ws)",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=None,
        help="Path to baseline model directory (default: models/baseline/)",
    )
    parser.add_argument(
        "--teardown", action="store_true",
        help="Delete endpoint and stop billing",
    )
    parser.add_argument(
        "--logs", action="store_true",
        help="Fetch deployment logs for debugging",
    )

    args = parser.parse_args()

    ml_client = get_ml_client(
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
    )

    if args.teardown:
        teardown(ml_client)
    elif args.logs:
        get_logs(ml_client)
    else:
        model = register_model(ml_client, args.model_dir)
        deploy_endpoint(ml_client, model)


if __name__ == "__main__":
    main()
