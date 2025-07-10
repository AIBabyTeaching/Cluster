# Code Overview

This document explains the main components in the repository and how they tie together for distributed language model training. The repository is organized around a set of small examples located under `labs/` and a minimal production-style workflow for fine-tuning TinyLlama.

## Common Utilities

`parallel_utils.py` contains helper functions that wrap `torch.distributed` initialization. The `init_distributed` function configures either NCCL or Gloo backends based on whether GPUs are available. `world_size()` and `is_main_process()` provide simple helpers for querying the distributed environment.

## Fine Tuning TinyLlama

`fine_tune_tinyllama.py` demonstrates how to fine‑tune the [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) model across multiple processes. The script loads the WikiText dataset, tokenizes it, and trains the model using `DistributedDataParallel`. A checkpoint is saved every 100 steps from rank 0 so that only the lead process writes files.

This example can run inside a Docker container using the scripts in the `scripts/` directory.

## Labs Directory

The `labs/` folder provides smaller self‑contained examples focused on different parallelism strategies:

- `simple_model/train_simple.py` — trains a text classifier on the AG News dataset using the HuggingFace `Trainer`. Launching with `torchrun` enables data parallelism.
- `fine_tuning/fine_tune.py` — fine‑tunes a causal language model. The script mirrors the standard HuggingFace trainer workflow and accepts a `--dry_run` flag for quick CPU‑only testing.
- `transfer_learning/transfer.py` — demonstrates transfer learning by training a classifier on the IMDB dataset starting from a pretrained encoder.
- `ragging/rag_example.py` — runs a basic Retrieval‑Augmented Generation (RAG) example on a small Wikipedia subset.

All lab scripts call `init_distributed()` and honor the `--dry_run` option for easy experimentation on a single machine.

## Scripts for Cluster Execution

The `scripts/` directory contains utilities to build and deploy a Docker image or to launch jobs via Slurm:

- `push_image.sh` builds the Docker image and pushes it to a registry.
- `deploy_nodes.sh` pulls the image on each node and launches `torchrun` with host networking so that all processes can rendezvous.
- `slurm_job.sh` shows how to submit a multi‑node job on a Slurm cluster using `torchrun`.

## Running Locally

Every training script accepts a `--dry_run` flag to perform a short pass on CPU. This is useful when testing code without a cluster. For example:

```bash
python labs/simple_model/train_simple.py --dry_run
```

## Further Reading

Refer to `README.md` in the repository root for high‑level setup instructions and for details on building Docker images for multi‑node training.
