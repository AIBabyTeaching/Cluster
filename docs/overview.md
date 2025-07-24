# Code Overview

This document explains the main components in the repository and how they fit together for distributed language model training on a Slurm cluster. Each directory contains a focused example that can be run on a single machine or scaled out across nodes.

## Common Utilities

`utils/parallel_utils.py` contains helper functions that wrap `torch.distributed` initialization. The `init_distributed` function configures either NCCL or Gloo backends based on whether GPUs are available. `world_size()` and `is_main_process()` provide simple helpers for querying the distributed environment.

`utils/Connect2Cluster.py` offers an interactive SSH client built on Paramiko. It keeps the terminal responsive by forwarding Ctrl+C and window resize events, making it easier to work on remote machines. The small `utils/checker.py` script prints cluster details such as the available partitions and current limits.

## Fine Tuning TinyLlama

`fine_tune_tinyllama.py` demonstrates how to fine‑tune the [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) model across multiple processes. The script downloads the WikiText dataset, tokenizes it, and runs the training loop under `DistributedDataParallel`. Rank 0 saves a checkpoint every 100 steps so that only the lead process writes files.


## Labs Directory

The `labs/` folder provides smaller self‑contained examples focused on different parallelism strategies:

- `simple_model/train_simple.py` — trains a text classifier on the AG News dataset using the HuggingFace `Trainer`. Launching with `torchrun` enables data parallelism and each process automatically receives a different data shard.
- `fine_tuning/fine_tune.py` — fine‑tunes a causal language model with a training loop similar to the official HuggingFace example. The optional `--dry_run` flag shortens the dataset so you can test the workflow on a laptop.
- `transfer_learning/transfer.py` — attaches a new classification head to a pretrained encoder and fine‑tunes it on the IMDB dataset.
- `ragging/rag_example.py` — demonstrates Retrieval‑Augmented Generation on a tiny Wikipedia subset and prints the retrieved passages for inspection.

All lab scripts call `init_distributed()` and honor the `--dry_run` option for easy experimentation on a single machine.

## Scripts for Cluster Execution

The `scripts/` directory contains helper tools for launching multi‑node jobs via Slurm:

- `slurm_job.sh` is a template Slurm submission script. It configures the number of nodes and tasks per node before invoking `torchrun` so that all processes join the same rendezvous.

## Running Locally

Every training script accepts a `--dry_run` flag to perform a short pass on CPU. This is useful when testing code without a cluster. For example:

```bash
python labs/simple_model/train_simple.py --dry_run
```

## Further Reading

Refer to `README.md` in the repository root for high‑level setup instructions on connecting to the cluster and preparing the Python environment.
