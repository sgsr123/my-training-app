# Vertex AI Python Training Application

This project contains a Python training application packaged as a source distribution
for use with Vertex AI prebuilt containers.

## Directory Structure
my-training-app/
├── setup.py               # Package configuration
├── trainer/               # Training code package
│   ├── init.py        # Package initialization
│   ├── task.py            # Main entry point for training
│   ├── model.py           # Model definition
│   └── utils.py           # Utility functions
└── scripts/               # Helper scripts
├── build_package.sh   # Script to build and upload the package
├── create_training_job.sh  # Script to create a training job with gcloud
└── create_training_job.py  # Script to create a training job with Python SDK