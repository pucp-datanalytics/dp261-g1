#!/usr/bin/env bash
# User-data de referencia para una EC2 Ubuntu/Amazon Linux con Docker.
# Reemplazar <...> antes de usar.
set -euo pipefail

AWS_REGION="<us-east-1>"
AWS_ACCOUNT_ID="<account-id>"
ECR_REPOSITORY="badbuy-api"
IMAGE_TAG="latest"
API_KEY="<api-key>"
ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}"

sudo yum update -y || sudo apt-get update -y
sudo yum install -y docker awscli || sudo apt-get install -y docker.io awscli
sudo systemctl enable docker
sudo systemctl start docker

aws ecr get-login-password --region "${AWS_REGION}" \
  | sudo docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

sudo docker pull "${ECR_URL}:${IMAGE_TAG}"
sudo docker rm -f badbuy-api || true
sudo docker run -d --name badbuy-api \
  -p 80:8000 \
  -e REQUIRE_API_KEY=true \
  -e API_KEY="${API_KEY}" \
  -e MODEL_THRESHOLD=0.70 \
  -e MODEL_VERSION="sprint5-final-threshold-0.70" \
  "${ECR_URL}:${IMAGE_TAG}"
