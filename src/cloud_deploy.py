"""
Cloud deployment module for the Glove Speed Tracker application.
Provides functionality to deploy the application to various cloud platforms.
"""

import os
import sys
import logging
import subprocess
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cloud_deploy')

class CloudDeployer:
    """
    Class for deploying the Glove Speed Tracker application to cloud platforms.
    """
    
    def __init__(self, project_dir=None):
        """
        Initialize the cloud deployer.
        
        Args:
            project_dir (str, optional): Path to the project directory
        """
        if project_dir is None:
            # Use the parent directory of the current file
            self.project_dir = str(Path(__file__).parent.parent.absolute())
        else:
            self.project_dir = project_dir
            
        logger.info(f"CloudDeployer initialized with project directory: {self.project_dir}")
        
    def check_prerequisites(self, target):
        """
        Check if the prerequisites for deployment are installed.
        
        Args:
            target (str): Deployment target (local, gcp-run, gcp-k8s, aws)
            
        Returns:
            bool: True if all prerequisites are installed, False otherwise
        """
        logger.info(f"Checking prerequisites for {target} deployment")
        
        # Check Docker installation
        try:
            subprocess.run(["docker", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Docker is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Docker is not installed")
            return False
        
        # Check target-specific prerequisites
        if target == "local":
            try:
                subprocess.run(["docker-compose", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logger.info("Docker Compose is installed")
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.error("Docker Compose is not installed")
                return False
                
        elif target in ["gcp-run", "gcp-k8s"]:
            try:
                subprocess.run(["gcloud", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logger.info("Google Cloud SDK is installed")
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.error("Google Cloud SDK is not installed")
                return False
                
            if target == "gcp-k8s":
                try:
                    subprocess.run(["kubectl", "version", "--client"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    logger.info("kubectl is installed")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    logger.error("kubectl is not installed")
                    return False
                    
        elif target == "aws":
            try:
                subprocess.run(["aws", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logger.info("AWS CLI is installed")
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.error("AWS CLI is not installed")
                return False
        
        logger.info(f"All prerequisites for {target} deployment are installed")
        return True
    
    def build_docker_image(self, image_name, image_tag="latest"):
        """
        Build a Docker image for the application.
        
        Args:
            image_name (str): Name of the Docker image
            image_tag (str, optional): Tag for the Docker image
            
        Returns:
            bool: True if the image was built successfully, False otherwise
        """
        logger.info(f"Building Docker image: {image_name}:{image_tag}")
        
        try:
            # Change to the project directory
            os.chdir(self.project_dir)
            
            # Build the Docker image
            subprocess.run(
                ["docker", "build", "-t", f"{image_name}:{image_tag}", "."],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info(f"Docker image {image_name}:{image_tag} built successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error building Docker image: {e}")
            logger.error(f"stderr: {e.stderr.decode()}")
            return False
    
    def deploy_local(self):
        """
        Deploy the application locally using Docker Compose.
        
        Returns:
            bool: True if the deployment was successful, False otherwise
        """
        logger.info("Deploying application locally using Docker Compose")
        
        try:
            # Change to the project directory
            os.chdir(self.project_dir)
            
            # Build and start the containers
            subprocess.run(
                ["docker-compose", "build"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            subprocess.run(
                ["docker-compose", "up", "-d"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info("Application deployed locally successfully")
            logger.info("The application is now running at: http://localhost:5000")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error deploying application locally: {e}")
            logger.error(f"stderr: {e.stderr.decode()}")
            return False
    
    def deploy_gcp_run(self, project_id, region="us-central1", image_name="glove-speed-tracker", image_tag="latest"):
        """
        Deploy the application to Google Cloud Run.
        
        Args:
            project_id (str): Google Cloud project ID
            region (str, optional): Google Cloud region
            image_name (str, optional): Name of the Docker image
            image_tag (str, optional): Tag for the Docker image
            
        Returns:
            bool: True if the deployment was successful, False otherwise
        """
        logger.info(f"Deploying application to Google Cloud Run in project {project_id}")
        
        try:
            # Change to the project directory
            os.chdir(self.project_dir)
            
            # Set the GCP project
            subprocess.run(
                ["gcloud", "config", "set", "project", project_id],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Build the Docker image
            image_path = f"gcr.io/{project_id}/{image_name}:{image_tag}"
            subprocess.run(
                ["docker", "build", "-t", image_path, "."],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Configure Docker to use gcloud as a credential helper
            subprocess.run(
                ["gcloud", "auth", "configure-docker", "-q"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Push the image to Google Container Registry
            subprocess.run(
                ["docker", "push", image_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Deploy to Cloud Run
            deploy_result = subprocess.run(
                [
                    "gcloud", "run", "deploy", image_name,
                    "--image", image_path,
                    "--platform", "managed",
                    "--region", region,
                    "--allow-unauthenticated",
                    "--memory", "2Gi",
                    "--cpu", "1",
                    "--port", "5000",
                    "--set-env-vars", "PORT=5000"
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Get the deployed URL
            service_url_result = subprocess.run(
                [
                    "gcloud", "run", "services", "describe", image_name,
                    "--platform", "managed",
                    "--region", region,
                    "--format", "value(status.url)"
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            service_url = service_url_result.stdout.decode().strip()
            
            logger.info("Application deployed to Google Cloud Run successfully")
            logger.info(f"The application is now running at: {service_url}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error deploying application to Google Cloud Run: {e}")
            logger.error(f"stderr: {e.stderr.decode()}")
            return False
    
    def deploy_gcp_k8s(self, project_id, region="us-central1", namespace="default", image_name="glove-speed-tracker", image_tag="latest"):
        """
        Deploy the application to Google Kubernetes Engine.
        
        Args:
            project_id (str): Google Cloud project ID
            region (str, optional): Google Cloud region
            namespace (str, optional): Kubernetes namespace
            image_name (str, optional): Name of the Docker image
            image_tag (str, optional): Tag for the Docker image
            
        Returns:
            bool: True if the deployment was successful, False otherwise
        """
        logger.info(f"Deploying application to Google Kubernetes Engine in project {project_id}")
        
        try:
            # Change to the project directory
            os.chdir(self.project_dir)
            
            # Set the GCP project
            subprocess.run(
                ["gcloud", "config", "set", "project", project_id],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Build the Docker image
            image_path = f"gcr.io/{project_id}/{image_name}:{image_tag}"
            subprocess.run(
                ["docker", "build", "-t", image_path, "."],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Configure Docker to use gcloud as a credential helper
            subprocess.run(
                ["gcloud", "auth", "configure-docker", "-q"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Push the image to Google Container Registry
            subprocess.run(
                ["docker", "push", image_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Update the Kubernetes deployment file with the correct image
            k8s_deployment_file = os.path.join(self.project_dir, "kubernetes", "deployment.yaml")
            with open(k8s_deployment_file, "r") as f:
                deployment_yaml = f.read()
            
            deployment_yaml = deployment_yaml.replace("gcr.io/PROJECT_ID/glove-speed-tracker:latest", image_path)
            
            with open(k8s_deployment_file, "w") as f:
                f.write(deployment_yaml)
            
            # Create namespace if it doesn't exist
            subprocess.run(
                ["kubectl", "create", "namespace", namespace, "--dry-run=client", "-o", "yaml"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Apply the Kubernetes configuration
            subprocess.run(
                ["kubectl", "apply", "-f", k8s_deployment_file, "-n", namespace],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for the deployment to be ready
            subprocess.run(
                ["kubectl", "rollout", "status", "deployment/glove-speed-tracker", "-n", namespace],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Get the external IP
            external_ip = None
            max_attempts = 30
            attempt = 0
            
            while external_ip is None and attempt < max_attempts:
                attempt += 1
                
                try:
                    external_ip_result = subprocess.run(
                        [
                            "kubectl", "get", "service", "glove-speed-tracker-service",
                            "-n", namespace,
                            "-o", "jsonpath={.status.loadBalancer.ingress[0].ip}"
                        ],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    external_ip = external_ip_result.stdout.decode().strip()
                    
                    if not external_ip:
                        external_ip = None
                        logger.info(f"Waiting for external IP (attempt {attempt}/{max_attempts})...")
                        import time
                        time.sleep(10)
                except subprocess.CalledProcessError:
                    logger.info(f"Waiting for external IP (attempt {attempt}/{max_attempts})...")
                    import time
                    time.sleep(10)
            
            if external_ip:
                logger.info("Application deployed to Google Kubernetes Engine successfully")
                logger.info(f"The application is now running at: http://{external_ip}")
                return True
            else:
                logger.error("Timed out waiting for external IP")
                return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Error deploying application to Google Kubernetes Engine: {e}")
            logger.error(f"stderr: {e.stderr.decode()}")
            return False
    
    def deploy_aws(self, region="us-east-1", image_name="glove-speed-tracker", image_tag="latest"):
        """
        Deploy the application to AWS.
        
        Args:
            region (str, optional): AWS region
            image_name (str, optional): Name of the Docker image
            image_tag (str, optional): Tag for the Docker image
            
        Returns:
            bool: True if the deployment was successful, False otherwise
        """
        logger.info(f"Deploying application to AWS in region {region}")
        
        # This is a placeholder for AWS deployment
        # In a real implementation, this would use the AWS CLI or SDK to deploy the application
        
        logger.warning("AWS deployment is not fully implemented yet")
        logger.info("Please refer to the documentation for manual AWS deployment steps")
        
        return False
    
    def deploy(self, target, **kwargs):
        """
        Deploy the application to the specified target.
        
        Args:
            target (str): Deployment target (local, gcp-run, gcp-k8s, aws)
            **kwargs: Additional arguments for the specific deployment target
            
        Returns:
            bool: True if the deployment was successful, False otherwise
        """
        logger.info(f"Deploying application to {target}")
        
        # Check prerequisites
        if not self.check_prerequisites(target):
            logger.error(f"Prerequisites for {target} deployment are not met")
            return False
        
        # Deploy to the specified target
        if target == "local":
            return self.deploy_local()
        elif target == "gcp-run":
            if "project_id" not in kwargs:
                logger.error("project_id is required for GCP Cloud Run deployment")
                return False
            
            return self.deploy_gcp_run(
                project_id=kwargs["project_id"],
                region=kwargs.get("region", "us-central1"),
                image_name=kwargs.get("image_name", "glove-speed-tracker"),
                image_tag=kwargs.get("image_tag", "latest")
            )
        elif target == "gcp-k8s":
            if "project_id" not in kwargs:
                logger.error("project_id is required for GCP Kubernetes Engine deployment")
                return False
            
            return self.deploy_gcp_k8s(
                project_id=kwargs["project_id"],
                region=kwargs.get("region", "us-central1"),
                namespace=kwargs.get("namespace", "default"),
                image_name=kwargs.get("image_name", "glove-speed-tracker"),
                image_tag=kwargs.get("image_tag", "latest")
            )
        elif target == "aws":
            return self.deploy_aws(
                region=kwargs.get("region", "us-east-1"),
                image_name=kwargs.get("image_name", "glove-speed-tracker"),
                image_tag=kwargs.get("image_tag", "latest")
            )
        else:
            logger.error(f"Unknown deployment target: {target}")
            return False

def main():
    """
    Main function to deploy the application from the command line.
    """
    parser = argparse.ArgumentParser(description="Deploy the Glove Speed Tracker application")
    parser.add_argument("--target", "-t", choices=["local", "gcp-run", "gcp-k8s", "aws"], default="local",
                        help="Deployment target (default: local)")
    parser.add_argument("--project-id", "-p", help="Google Cloud project ID (required for GCP deployments)")
    parser.add_argument("--region", "-r", help="Cloud region (default depends on target)")
    parser.add_argument("--namespace", "-n", default="default", help="Kubernetes namespace (default: default)")
    parser.add_argument("--image-name", "-i", default="glove-speed-tracker", help="Docker image name (default: glove-speed-tracker)")
    parser.add_argument("--image-tag", "-v", default="latest", help="Image version/tag (default: latest)")
    
    args = parser.parse_args()
    
    # Check required arguments
    if args.target in ["gcp-run", "gcp-k8s"] and not args.project_id:
        parser.error(f"--project-id is required for {args.target} deployment")
    
    # Create deployer
    deployer = CloudDeployer()
    
    # Deploy
    success = deployer.deploy(
        target=args.target,
        project_id=args.project_id,
        region=args.region,
        namespace=args.namespace,
        image_name=args.image_name,
        image_tag=args.image_tag
    )
    
    if success:
        logger.info("Deployment completed successfully")
        return 0
    else:
        logger.error("Deployment failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
