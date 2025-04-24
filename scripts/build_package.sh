#!/bin/bash
# Script to build the Python source distribution and upload to GCS

# Check for required variables
if [ -z "$BUCKET_NAME" ]; then
    echo "Error: BUCKET_NAME environment variable not set"
    echo "Usage: BUCKET_NAME=your-bucket VERSION=0.1 ./build_package.sh"
    exit 1
fi

# Set default values
VERSION=${VERSION:-"0.1"}
PROJECT_ID=$(gcloud config list --format 'value(core.project)')
REGION=${REGION:-"us-central1"}
PACKAGE_DIR=${PACKAGE_DIR:-"packages"}
GCS_PATH="gs://${BUCKET_NAME}/${PACKAGE_DIR}/trainer-${VERSION}.tar.gz"

echo "Building package version ${VERSION}"

# Navigate to directory containing setup.py
cd "$(dirname "$0")/.."

# Build the distribution package
python setup.py sdist --formats=gztar

# Ensure GCS bucket exists
gsutil ls -b "gs://${BUCKET_NAME}" > /dev/null || gsutil mb -p ${PROJECT_ID} -l ${REGION} "gs://${BUCKET_NAME}"

# Upload to Google Cloud Storage
gsutil cp dist/trainer-${VERSION}.tar.gz ${GCS_PATH}

echo "Package uploaded to: ${GCS_PATH}"
echo "You can use this path in your training job configuration."