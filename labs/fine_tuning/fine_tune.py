"""Fine-tune a causal language model with optional distributed training."""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from parallel_utils import init_distributed, is_main_process


def main():
    """Fine-tune GPT-style models on a text corpus.

    This script mirrors the HuggingFace trainer example but integrates our
    distributed utilities.  Use ``--dry_run`` for a tiny dataset slice when
    experimenting locally.
    """

    parser = argparse.ArgumentParser(description="Fine-tune a language model with parallelism")
    parser.add_argument("--model", default="gpt2", help="Base model")
    parser.add_argument("--dataset", default="wikitext", help="Dataset name")
    parser.add_argument("--subset", default="wikitext-2-raw-v1", help="Dataset subset")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", default="./finetuned")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=os.getenv("LOCAL_RANK", 0),
        help="Provided by torchrun",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run only a few samples for quick debugging",
    )
    args = parser.parse_args()

    init_distributed(args.local_rank)

    # Load the dataset and tokenizer. GPT2 models require a padding token for batching.
    dataset = load_dataset(args.dataset, args.subset)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

    # Tokenize the entire dataset upfront
    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format("torch")

    # Optionally take a very small slice for quick testing
    train_ds = tokenized["train"]
    if args.dry_run:
        train_ds = train_ds.select(range(64))

    model = AutoModelForCausalLM.from_pretrained(args.model)

    # Configure the HuggingFace Trainer. ``gradient_accumulation_steps`` can be
    # increased to simulate larger batch sizes when compute resources are limited.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=5e-5,
        evaluation_strategy="no",
        logging_steps=10,
        gradient_accumulation_steps=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )

    trainer.train()
    if is_main_process():
        trainer.save_model(args.output_dir)


if __name__ == "__main__":
    # Works on CPU for a dry run or across multiple processes with torchrun
    main()

