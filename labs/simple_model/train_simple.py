"""Minimal text classification example using data parallelism."""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from parallel_utils import init_distributed, is_main_process

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

    init_distributed(args.local_rank)

    # Load a small text classification dataset and tokenizer
    dataset = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

    # Preprocess the dataset. HuggingFace datasets will automatically use
    # multiprocessing here when available.
    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=4)

    # TrainingArguments control the training loop and distributed setup
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_dir="./logs",
        logging_steps=10,
        dataloader_drop_last=True,
    )

    # Optionally shrink datasets for a fast dry run
    train_ds = tokenized["train"]
    eval_ds = tokenized["test"]
    if args.dry_run:
        train_ds = train_ds.select(range(64))
        eval_ds = eval_ds.select(range(64))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # Perform training. Only the main process will save the final model to
    # avoid race conditions when running with multiple ranks.
    trainer.train()
    if is_main_process():
        trainer.save_model(args.output_dir)

if __name__ == "__main__":
    # When executed directly this will run on a single process.  Use torchrun
    # for distributed multi-process training.
    main()

