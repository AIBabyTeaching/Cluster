"""Train a text classifier with data parallelism.

This lab script illustrates the bare minimum needed to launch a distributed
training job with the HuggingFace ``Trainer``.  It uses the AG News dataset and
the ``distilroberta-base`` model by default.  When executed with ``torchrun``
each process will receive a different chunk of data and gradients will be
aggregated automatically.  Pass ``--dry_run`` to run a tiny subset on CPU for
teaching or debugging.
"""

import argparse
import os
import logging

import torch
# ----------------------------------------------------------------------------
# Monkey-patch Accelerate’s CPU backend so it won’t call torch.cpu.set_device()
if not torch.cuda.is_available():
    try:
        import torch.cpu
        torch.cpu.set_device = lambda device: None
    except ImportError:
        pass
# ----------------------------------------------------------------------------

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from utils.parallel_utils import init_distributed, is_main_process

def main():
    """Entry point for training.

    Launch this script with ``torchrun`` to enable distributed data
    parallelism.  Use the ``--dry_run`` flag to execute quickly on a
    workstation without multiple processes or GPUs.
    """
    parser = argparse.ArgumentParser(description="Train a simple classification model with data parallelism")
    parser.add_argument("--model", default="distilroberta-base", help="Model checkpoint")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--output_dir", default="./model_output", help="Save directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Per device batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=os.getenv("LOCAL_RANK", 0),
        help="Provided by torchrun for distributed execution",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run on a small subset of the data for quick local testing",
    )
    args = parser.parse_args()

    # Set up torch.distributed so Trainer will aggregate gradients.
    init_distributed(args.local_rank)

    # ------------------ per-rank logging for debugging ------------------
    import torch.distributed as dist
    if dist.is_initialized():
        rank    = dist.get_rank()
        world   = dist.get_world_size()
        backend = dist.get_backend()
    else:
        rank, world, backend = 0, 1, "none"
    hostname = os.uname().nodename
    print(f"[Rank {rank}/{world} | backend={backend}] PID={os.getpid()} on {hostname}")
    # --------------------------------------------------------------------

    # Use a writable cache directory to avoid permission errors:
    cache_root = os.environ.get("HF_HOME", os.path.join(os.getcwd(), ".hf_cache"))
    os.environ["HF_HOME"] = cache_root
    os.makedirs(cache_root, exist_ok=True)

    # Debug: announce dataset loading
    print(f"[Rank {rank}] Loading AG News dataset")
    dataset = load_dataset("ag_news", cache_dir=os.path.join(cache_root, "datasets"))
    if args.dry_run:
        # only run map() over a tiny subset
        dataset["train"] = dataset["train"].select(range(64))
        dataset["test"]  = dataset["test"].select(range(64))
    print(f"[Rank {rank}] Tokenizing dataset")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=os.path.join(cache_root, "models")
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Debug: show dataset sizes
    print(f"[Rank {rank}] train size = {len(tokenized['train'])}, eval size = {len(tokenized['test'])}")

    num_labels = dataset["train"].features["label"].num_classes
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        cache_dir=os.path.join(cache_root, "models"),
        local_files_only=True
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_dir="./logs",
        logging_steps=1,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,      # remove warning
        log_level="debug",                   # verbose logs
        max_steps=5, # SHOULD BE like 1000
    )

    # Debug: show TrainingArguments
    print(f"[Rank {rank}] TrainingArguments: {training_args}")

    train_ds = tokenized["train"]
    eval_ds  = tokenized["test"]
    if args.dry_run:
        train_ds = train_ds.select(range(64))
        eval_ds  = eval_ds.select(range(64))

    # Debug: sample a batch
    from torch.utils.data import DataLoader
    dl = DataLoader(train_ds, batch_size=args.batch_size)
    batch = next(iter(dl))
    print(f"[Rank {rank}] Sample batch shapes: input_ids={batch['input_ids'].shape}, labels={batch['labels'].shape}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    print(f"[Rank {rank}] Starting trainer.train()")
    trainer.train()
    print(f"[Rank {rank}] trainer.train() completed")

    if is_main_process():
        print(f"[Rank {rank}] Saving model to {args.output_dir}")
        trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()

# RUNNING BY THE FOLLOWING
'''

unset TRANSFORMERS_OFFLINE HUGGINGFACE_HUB_OFFLINE HF_DATASETS_OFFLINE
python - <<EOF
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# point HF_HOME at your shared cache
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

# download the AG News dataset
load_dataset("ag_news", cache_dir=os.path.join(os.environ["HF_HOME"], "datasets"))

# download distilroberta-base (model weights + index JSON)
AutoTokenizer.from_pretrained(
    "distilroberta-base",
    cache_dir=os.path.join(os.environ["HF_HOME"], "models")
)
AutoModelForSequenceClassification.from_pretrained(
    "distilroberta-base",
    num_labels=4,
    cache_dir=os.path.join(os.environ["HF_HOME"], "models")
)
EOF


export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME

ls ~/.cache/huggingface/models/distilroberta-base

export TRANSFORMERS_OFFLINE=1
export HUGGINGFACE_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1



# 1) grab a 20‐node interactive block
salloc \
  --nodes=20 \
  --ntasks-per-node=1 \
  --cpus-per-task=2 \
  --time=00:05:00 \
  --partition=parallel \
  --exclusive


source ~/llamaenv_local/bin/activate
cd ~/mrmito/project

export HF_HOME=$HOME/.cache/huggingface
export PYTHONPATH=$PWD:$PYTHONPATH

# pick a random free port each time
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=$((20000 + RANDOM % 10000))

srun \
  --nodes=$SLURM_NNODES \
  --ntasks-per-node=1 \
  --cpus-per-task=2 \
    --kill-on-bad-exit=0 \
  --unbuffered \
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=2 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv-id=$SLURM_JOB_ID \
    --rdzv-conf timeout=600 \
    labs/simple_model/train_simple.py --dry_run



'''

"""
start replacing slurm
"""