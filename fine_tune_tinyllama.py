"""Minimal TinyLlama fine-tuning script for distributed CPU clusters."""

import argparse
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def init_distributed() -> int:
    """Initialize ``torch.distributed`` with the gloo backend."""
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    dist.init_process_group("gloo")
    return local_rank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)), help="Provided by torchrun"
    )
    args = parser.parse_args()

    local_rank = init_distributed()
    torch.manual_seed(0)

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch")

    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", output_attentions=True
    )
    model = DDP(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0 and dist.get_rank() == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")
            if step % 100 == 0 and dist.get_rank() == 0:
                ckpt = f"checkpoint_epoch{epoch}_step{step}.pt"
                torch.save(model.module.state_dict(), ckpt)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
