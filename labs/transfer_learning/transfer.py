"""Demonstrates transfer learning on the IMDB dataset.

The script adds a classification head on top of a pretrained encoder and fine
tunes it on movie reviews. It reuses the same distributed setup helpers as the
other labs so that you can scale out the experiment with ``torchrun``.
"""

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
    """Fine-tune a classification head on top of a pretrained encoder."""

    parser = argparse.ArgumentParser(description="Transfer learning example")
    parser.add_argument("--base_model", default="distilbert-base-uncased")
    parser.add_argument("--dataset", default="imdb")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output_dir", default="./transfer_model")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=os.getenv("LOCAL_RANK", 0),
        help="Assigned by torchrun",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run a quick pass with a handful of samples",
    )
    args = parser.parse_args()

    # Set up the process group so that the Trainer will synchronize gradients.
    init_distributed(args.local_rank)

    # Load the movie review dataset and tokenizer
    dataset = load_dataset(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

    # Preprocess text to tensor format
    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=8,
        evaluation_strategy='epoch',
        num_train_epochs=args.epochs,
    )

    # Limit the dataset size when performing a dry run
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
    # ``Trainer`` manages the training loop and distributed synchronization.
    trainer.train()
    if is_main_process():
        trainer.save_model(args.output_dir)


if __name__ == "__main__":
    # Execute with torchrun on the cluster or run locally for testing
    main()

