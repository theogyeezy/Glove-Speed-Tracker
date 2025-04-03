#!/bin/bash

# Deployment script for Glove Speed Tracker application
# This script helps deploy the application to different cloud environments

# Set default values
DEPLOY_TARGET="local"
IMAGE_NAME="glove-speed-tracker"
IMAGE_TAG="latest"
GCP_PROJECT=""
GCP_REGION="us-central1"
K8S_NAMESPACE="default"

# Display help message
show_help() {
    echo "Usage: ./deploy.sh [options]"
    echo ""
    echo "Options:"
    echo "  -t, --target     Deployment target (local, gcp-run, gcp-k8s, aws)"
    echo "  -p, --project    GCP project ID (required for GCP deployments)"
    echo "  -r, --region     Cloud region (default: us-central1 for GCP)"
    echo "  -i, --image      Docker image name (default: glove-speed-tracker)"
    echo "  -v, --version    Image version/tag (default: latest)"
    echo "  -n, --namespace  Kubernetes namespace (default: default)"
    echo "  -h, --help       Display this help message"
    echo ""
    echo "Examples:"
    echo "  ./deploy.sh --target local"
    echo "  ./deploy.sh --target gcp-run --project my-gcp-project"
    echo "  ./deploy.sh --target gcp-k8s --project my-gcp-project --namespace production"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -t|--target)
            DEPLOY_TARGET="$2"
            shift
            shift
            ;;
        -p|--project)
            GCP_PROJECT="$2"
            shift
            shift
            ;;
        -r|--region)
            GCP_REGION="$2"
            shift
            shift
            ;;
        -i|--image)
            IMAGE_NAME="$2"
            shift
            shift
            ;;
        -v|--version)
            IMAGE_TAG="$2"
            shift
            shift
            ;;
        -n|--namespace)
            K8S_NAMESPACE="$2"
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

# Validate required parameters based on deployment target
if [[ "$DEPLOY_TARGET" == "gcp-run" || "$DEPLOY_TARGET" == "gcp-k8s" ]]; then
    if [[ -z "$GCP_PROJECT" ]]; then
        echo "Error: GCP project ID is required for GCP deployments"
        echo "Use --project to specify your GCP project ID"
        exit 1
    fi
fi

# Function to deploy locally using Docker Compose
deploy_local() {
    echo "Deploying Glove Speed Tracker locally using Docker Compose..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker is not installed"
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        echo "Error: Docker Compose is not installed"
        exit 1
    fi
    
    # Build and start the containers
    docker-compose build
    docker-compose up -d
    
    echo "Deployment completed successfully!"
    echo "The application is now running at: http://localhost:5000"
}

# Function to deploy to Google Cloud Run
deploy_gcp_run() {
    echo "Deploying Glove Speed Tracker to Google Cloud Run..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        echo "Error: Google Cloud SDK (gcloud) is not installed"
        exit 1
    fi
    
    # Check if user is authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
        echo "Error: Not authenticated with gcloud"
        echo "Please run 'gcloud auth login' first"
        exit 1
    fi
    
    # Set the GCP project
    gcloud config set project "$GCP_PROJECT"
    
    # Build the Docker image
    echo "Building Docker image..."
    docker build -t "gcr.io/$GCP_PROJECT/$IMAGE_NAME:$IMAGE_TAG" .
    
    # Push the image to Google Container Registry
    echo "Pushing image to Google Container Registry..."
    gcloud auth configure-docker -q
    docker push "gcr.io/$GCP_PROJECT/$IMAGE_NAME:$IMAGE_TAG"
    
    # Deploy to Cloud Run
    echo "Deploying to Cloud Run..."
    gcloud run deploy "$IMAGE_NAME" \
        --image "gcr.io/$GCP_PROJECT/$IMAGE_NAME:$IMAGE_TAG" \
        --platform managed \
        --region "$GCP_REGION" \
        --allow-unauthenticated \
        --memory 2Gi \
        --cpu 1 \
        --port 5000 \
        --set-env-vars=PORT=5000
    
    # Get the deployed URL
    SERVICE_URL=$(gcloud run services describe "$IMAGE_NAME" --platform managed --region "$GCP_REGION" --format="value(status.url)")
    
    echo "Deployment completed successfully!"
    echo "The application is now running at: $SERVICE_URL"
}

# Function to deploy to Google Kubernetes Engine
deploy_gcp_k8s() {
    echo "Deploying Glove Speed Tracker to Google Kubernetes Engine..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        echo "Error: kubectl is not installed"
        exit 1
    fi
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        echo "Error: Google Cloud SDK (gcloud) is not installed"
        exit 1
    fi
    
    # Set the GCP project
    gcloud config set project "$GCP_PROJECT"
    
    # Build the Docker image
    echo "Building Docker image..."
    docker build -t "gcr.io/$GCP_PROJECT/$IMAGE_NAME:$IMAGE_TAG" .
    
    # Push the image to Google Container Registry
    echo "Pushing image to Google Container Registry..."
    gcloud auth configure-docker -q
    docker push "gcr.io/$GCP_PROJECT/$IMAGE_NAME:$IMAGE_TAG"
    
    # Update the Kubernetes deployment file with the correct image
    echo "Updating Kubernetes deployment configuration..."
    sed -i "s|gcr.io/PROJECT_ID/glove-speed-tracker:latest|gcr.io/$GCP_PROJECT/$IMAGE_NAME:$IMAGE_TAG|g" kubernetes/deployment.yaml
    
    # Create namespace if it doesn't exist
    kubectl get namespace "$K8S_NAMESPACE" &> /dev/null || kubectl create namespace "$K8S_NAMESPACE"
    
    # Apply the Kubernetes configuration
    echo "Applying Kubernetes configuration..."
    kubectl apply -f kubernetes/deployment.yaml -n "$K8S_NAMESPACE"
    
    # Wait for the deployment to be ready
    echo "Waiting for deployment to be ready..."
    kubectl rollout status deployment/glove-speed-tracker -n "$K8S_NAMESPACE"
    
    # Get the external IP
    echo "Getting service external IP..."
    EXTERNAL_IP=""
    while [ -z "$EXTERNAL_IP" ]; do
        EXTERNAL_IP=$(kubectl get service glove-speed-tracker-service -n "$K8S_NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        if [ -z "$EXTERNAL_IP" ]; then
            echo "Waiting for external IP..."
            sleep 10
        fi
    done
    
    echo "Deployment completed successfully!"
    echo "The application is now running at: http://$EXTERNAL_IP"
}

# Function to deploy to AWS (placeholder)
deploy_aws() {
    echo "Deploying Glove Speed Tracker to AWS..."
    echo "AWS deployment is not fully implemented yet."
    echo "Please refer to the documentation for manual AWS deployment steps."
}

# Execute deployment based on target
case "$DEPLOY_TARGET" in
    local)
        deploy_local
        ;;
    gcp-run)
        deploy_gcp_run
        ;;
    gcp-k8s)
        deploy_gcp_k8s
        ;;
    aws)
        deploy_aws
        ;;
    *)
        echo "Error: Unknown deployment target: $DEPLOY_TARGET"
        show_help
        exit 1
        ;;
esac
