#!/usr/bin/env bash
set -euo pipefail

: "${NUM_NODES:?}"    # total number of nodes
: "${PROCS_PER_NODE:?}" # processes per node
: "${MASTER_IP:?}"      # IP address of the master node
: "${BATCH:?}"         # batch size per process
: "${EPOCHS:?}"        # number of epochs

IMAGE="my.registry:5000/torch-tinyllama:cpu"

# Pull the latest image from the registry
docker pull "$IMAGE"

# Stop and remove an existing container if present
if docker ps -a --format '{{.Names}}' | grep -q '^torch_node$'; then
    docker stop torch_node
    docker rm torch_node
fi

# Launch distributed training using host networking
docker run -d --name torch_node --network host "$IMAGE" \
  torchrun \
  --nnodes="$NUM_NODES" \
  --nproc_per_node="$PROCS_PER_NODE" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_IP:29500" \
  fine_tune_tinyllama.py \
  --batch_size "$BATCH" \
  --epochs "$EPOCHS"
