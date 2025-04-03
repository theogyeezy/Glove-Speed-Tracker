#!/bin/bash

# Script to deploy the Glove Speed Tracker application to Google Cloud App Engine
# This creates a permanent website with a fixed URL

# Set default values
PROJECT_ID=""
VERSION="v1"
REGION="us-central"

# Display help message
show_help() {
    echo "Usage: ./deploy_app_engine.sh [options]"
    echo ""
    echo "Options:"
    echo "  -p, --project    GCP project ID (required)"
    echo "  -v, --version    App version (default: v1)"
    echo "  -r, --region     App Engine region (default: us-central)"
    echo "  -h, --help       Display this help message"
    echo ""
    echo "Examples:"
    echo "  ./deploy_app_engine.sh --project my-gcp-project"
    echo "  ./deploy_app_engine.sh --project my-gcp-project --version v2 --region us-east1"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -p|--project)
            PROJECT_ID="$2"
            shift
            shift
            ;;
        -v|--version)
            VERSION="$2"
            shift
            shift
            ;;
        -r|--region)
            REGION="$2"
            shift
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$PROJECT_ID" ]]; then
    echo "Error: GCP project ID is required"
    echo "Use --project to specify your GCP project ID"
    show_help
    exit 1
fi

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: Google Cloud SDK (gcloud) is not installed"
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo "Error: Not authenticated with gcloud"
    echo "Please run 'gcloud auth login' first"
    exit 1
fi

# Set the GCP project
echo "Setting GCP project to: $PROJECT_ID"
gcloud config set project "$PROJECT_ID"

# Check if App Engine application exists
if ! gcloud app describe &> /dev/null; then
    echo "Creating App Engine application in region: $REGION"
    gcloud app create --region="$REGION"
else
    echo "App Engine application already exists"
fi

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable appengine.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Deploy the application
echo "Deploying Glove Speed Tracker to App Engine..."
gcloud app deploy app.yaml --version="$VERSION" --quiet

# Get the deployed URL
APP_URL=$(gcloud app browse --no-launch-browser)

echo "Deployment completed successfully!"
echo "The application is now running at: $APP_URL"
echo ""
echo "To view logs, run: gcloud app logs tail"
echo "To open the application in your browser, run: gcloud app browse"
