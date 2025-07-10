# LLM Parallelism Labs

This repository contains example code for training and deploying small language models with different parallelism strategies. The labs are designed for a cluster environment (CPU-only by default) and focus on data parallelism, model fine-tuning, transfer learning, and retrieval-augmented generation (RAG).

The code relies on **torch.distributed** and the HuggingFace `Trainer` to demonstrate four key concepts:

See [docs/overview.md](docs/overview.md) for a guided tour of the codebase.

1. **Data Parallelism** – replicate the model across devices and split batches of data.
2. **Model Parallelism** – partition model layers across devices.
3. **Pipeline Parallelism** – sequence model stages across processes.
4. **Tensor Parallelism** – shard individual weight matrices for extreme scale.

## Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Launch multi-node jobs with `torchrun` or `deepspeed`. Example for a CPU-only cluster with two processes on one node:

```bash
torchrun --nnodes 1 --nproc_per_node 2 labs/simple_model/train_simple.py --epochs 1
```

### Local Dry Run

All training scripts accept a `--dry_run` flag which processes only a handful of
samples and runs on CPU. This is useful when experimenting without a cluster:

```bash
python labs/simple_model/train_simple.py --dry_run
```

### Cluster Launch

On clusters using a workload manager like **Slurm**, jobs can be submitted with
``torchrun`` inside a batch script.  A template is provided in
`scripts/slurm_job.sh`:

```bash
sbatch scripts/slurm_job.sh
```

## Labs Overview

### Simple Model
Train a text classifier on the AG News dataset using data parallelism:

```bash
torchrun --nproc_per_node 2 labs/simple_model/train_simple.py --epochs 1
```

### Fine Tuning
Fine-tune a causal language model on WikiText:

```bash
torchrun --nproc_per_node 2 labs/fine_tuning/fine_tune.py --epochs 1
```

### Transfer Learning
Continue training a sequence classifier on the IMDB dataset:

```bash
torchrun --nproc_per_node 2 labs/transfer_learning/transfer.py --epochs 1
```

### Retrieval-Augmented Generation
Run a basic RAG example:

```bash
python labs/ragging/rag_example.py --query "What is deep learning?"
```

These scripts are minimal and intended for instructional purposes. Adjust batch sizes and epochs to fit your cluster resources.

## Docker-Based Multi-Node Training

The repository includes utilities to build a container image and launch
`fine_tune_tinyllama.py` across multiple hosts. First start a local Docker
registry:

```bash
docker run -d -p 5000:5000 --name registry registry:2
```

Build and push the image:

```bash
./scripts/push_image.sh
```

On each node set the environment variables and deploy:

```bash
export MASTER_IP=<ip-of-node0>
export NUM_NODES=3
export PROCS_PER_NODE=4
export BATCH=1
export EPOCHS=1
./scripts/deploy_nodes.sh
```

`deploy_nodes.sh` runs `torchrun` inside the container with `--network host` so
the rendezvous uses the host IP/port, mirroring
[PyTorch's multi-node example](https://pytorch.org/docs/stable/elastic/run.html#distributed-running).
