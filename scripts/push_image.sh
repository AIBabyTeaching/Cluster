#!/usr/bin/env bash
set -euo pipefail

IMAGE="torch-tinyllama:cpu"
REGISTRY_IMAGE="my.registry:5000/torch-tinyllama:cpu"

# Build the Docker image
Dockerfile_dir="$(dirname "$(realpath "$0")")/.."
cd "$Dockerfile_dir"
docker build -t "$IMAGE" -f Dockerfile .

docker tag "$IMAGE" "$REGISTRY_IMAGE"
docker push "$REGISTRY_IMAGE"
