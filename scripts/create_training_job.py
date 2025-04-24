"""Create a Vertex AI training job using the Vertex AI SDK."""

import argparse
import os
from google.cloud import aiplatform

def create_training_job(
    project_id,
    location,
    display_name,
    python_package_uri,
    python_module,
    container_uri,
    args_list,
    machine_type="n1-standard-4",
    replica_count=1,
    service_account=None,
):
    """Create a Vertex AI custom training job using a prebuilt container.
    
    Args:
        project_id: Google Cloud project ID
        location: Google Cloud region
        display_name: Display name for the training job
        python_package_uri: GCS URI to the Python package
        python_module: Python module to run
        container_uri: URI of the prebuilt container
        args_list: List of arguments to pass to the Python module
        machine_type: Compute instance type
        replica_count: Number of worker replicas
        service_account: Service account email address
    """
    # Initialize the Vertex AI SDK
    aiplatform.init(project=project_id, location=location)
    
    # Create custom training job
    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=display_name,
        python_package_gcs_uri=python_package_uri,
        python_module_name=python_module,
        container_uri=container_uri,
    )
    
    # Get a GCS staging location for the job
    staging_bucket = f"gs://{args_list.get('bucket_name', project_id + '-staging')}"
    
    # Start the training job
    job.run(
        args=args_list,
        replica_count=replica_count,
        machine_type=machine_type,
        service_account=service_account,
        staging_bucket=staging_bucket,
        sync=True,  # Set to False for asynchronous execution
    )
    
    print(f"Training job completed: {display_name}")

def main():
    parser = argparse.ArgumentParser(description="Create a Vertex AI training job")
    
    # Required arguments
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
    parser.add_argument("--bucket-name", required=True, help="Cloud Storage bucket name")
    parser.add_argument("--package-uri", required=True, help="GCS URI to the Python package")
    
    # Optional arguments
    parser.add_argument("--region", default="us-central1", help="Google Cloud region")
    parser.add_argument("--display-name", default=f"custom-training-{os.getpid()}", help="Display name for the job")
    parser.add_argument("--python-module", default="trainer.task", help="Python module to run")
    parser.add_argument("--machine-type", default="n1-standard-4", help="Machine type")
    parser.add_argument("--container-uri", default="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-8:latest", 
                        help="Prebuilt container URI")
    parser.add_argument("--target-column", default="target", help="Target column in the dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    
    args = parser.parse_args()
    
    # Prepare arguments for the training job
    output_dir = f"gs://{args.bucket_name}/models/{args.display_name}"
    
    training_args = {
        "model_dir": output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "target_column": args.target_column,
        "bucket_name": args.bucket_name,
    }
    
    # Convert dictionary to list of argument strings
    args_list = [f"--{key}={value}" for key, value in training_args.items()]
    
    # Create and run the training job
    create_training_job(
        project_id=args.project_id,
        location=args.region,
        display_name=args.display_name,
        python_package_uri=args.package_uri,
        python_module=args.python_module,
        container_uri=args.container_uri,
        args_list=args_list,
        machine_type=args.machine_type,
    )

if __name__ == "__main__":
    main()