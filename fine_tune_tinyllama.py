"""Fine-tune TinyLlama on a CPU cluster.

This script intentionally keeps the training loop explicit so students can see
how ``DistributedDataParallel`` is wired up under the hood.  It tokenizes the
WikiText dataset, shuffles it with a ``DistributedSampler`` and saves periodic
checkpoints from rank 0 so multiple workers do not contend when writing files.
Launch with ``torchrun`` to enable multi-process training or run directly for a
local dry run.
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def init_distributed() -> int:
    """Initialize ``torch.distributed`` for CPU-only training.

    For simplicity we always use the ``gloo`` backend here which works on any
    hardware. The returned ``local_rank`` can be used to set the current device
    when GPUs are available. ``torchrun`` supplies ``LOCAL_RANK`` and other
    environment variables required by ``DistributedDataParallel``.
    """
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

    # Load tokenizer and dataset.  ``pad_token`` must be set so sequences can be
    # padded to equal length for batching.
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    # Tokenization is performed up front so the DataLoader can directly yield
    # tensors. ``remove_columns`` drops the original text to save memory.
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch")

    # ``DistributedSampler`` ensures each process receives a unique slice of the
    # dataset.  Shuffling is handled internally when ``set_epoch`` is called.
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # Wrap the model with ``DistributedDataParallel`` so gradients are reduced
    # across all processes during ``backward``.
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", output_attentions=True
    )
    model = DDP(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(args.epochs):
        # Calling ``set_epoch`` reshuffles the sampler deterministically so that
        # each epoch sees a different ordering while keeping processes in sync.
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
                # Only rank 0 writes checkpoints to avoid file system races.
                ckpt = f"checkpoint_epoch{epoch}_step{step}.pt"
                torch.save(model.module.state_dict(), ckpt)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
