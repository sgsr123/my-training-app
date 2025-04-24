#!/bin/bash
# Script to create a Vertex AI training job using a prebuilt container

# Check for required variables
if [ -z "$BUCKET_NAME" ]; then
    echo "Error: BUCKET_NAME environment variable not set"
    echo "Usage: BUCKET_NAME=your-bucket VERSION=0.1 REGION=us-central1 ./create_training_job.sh"
    exit 1
fi

# Set default values
VERSION=${VERSION:-"0.1"}
REGION=${REGION:-"us-central1"}
PROJECT_ID=$(gcloud config list --format 'value(core.project)')
PACKAGE_DIR=${PACKAGE_DIR:-"packages"}
JOB_NAME="custom-training-job-$(date +%Y%m%d-%H%M%S)"
MACHINE_TYPE=${MACHINE_TYPE:-"n1-standard-4"}
PYTHON_PACKAGE_URI="gs://${BUCKET_NAME}/${PACKAGE_DIR}/trainer-${VERSION}.tar.gz"
PYTHON_MODULE="trainer.task"
OUTPUT_DIR="gs://${BUCKET_NAME}/models/${JOB_NAME}"

# Choose the container image based on the framework and version
CONTAINER_IMAGE="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-8:latest"

echo "Creating training job: ${JOB_NAME}"
echo "Using package: ${PYTHON_PACKAGE_URI}"
echo "Output directory: ${OUTPUT_DIR}"

# Create the training job
gcloud ai custom-jobs create \
    --project=${PROJECT_ID} \
    --region=${REGION} \
    --display-name=${JOB_NAME} \
    --python-package-uris=${PYTHON_PACKAGE_URI} \
    --worker-pool-spec=machine-type=${MACHINE_TYPE},replica-count=1,executor-image-uri=${CONTAINER_IMAGE},python-module=${PYTHON_MODULE} \
    --args="--model-dir=${OUTPUT_DIR},--epochs=10,--batch-size=32,--target-column=target"

echo "Job submitted. Monitor progress in the Google Cloud Console."
echo "https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"