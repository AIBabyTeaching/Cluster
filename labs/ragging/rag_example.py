"""Simple Retrieval-Augmented Generation (RAG) demonstration.

This example fetches a small subset of Wikipedia passages and shows how the RAG
model can incorporate retrieved documents into the generation process. It also
supports a ``--dry_run`` flag to limit the number of documents for a quick
classroom demo.
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import RagRetriever, RagTokenizer, RagSequenceForGeneration

from parallel_utils import init_distributed, is_main_process


def main():
    """Run a small RAG example using a Wikipedia corpus."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_name", default="exact")
    parser.add_argument("--model_name", default="facebook/rag-sequence-base")
    parser.add_argument("--dataset", default="wiki_dpr")
    parser.add_argument("--subset", default="psgs_w100.multiset.small")
    parser.add_argument("--query", default="What is deep learning?")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=os.getenv("LOCAL_RANK", 0),
        help="Used by torchrun",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load fewer documents for quick testing",
    )
    args = parser.parse_args()

    init_distributed(args.local_rank)

    # Restrict the dataset when running a dry run to speed things up. This keeps
    # the demo lightweight while still exercising the retrieval mechanics.
    split = "train[:50]" if args.dry_run else "train[:500]"
    dataset = load_dataset(args.dataset, args.subset, split=split)
    tokenizer = RagTokenizer.from_pretrained(args.model_name)
    retriever = RagRetriever.from_pretrained(
        args.model_name, index_name=args.index_name, passages_path=None
    )
    model = RagSequenceForGeneration.from_pretrained(
        args.model_name, retriever=retriever
    )

    inputs = tokenizer(args.query, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(**inputs)
    if is_main_process():
        # Decode and print the generated answer only once from rank 0.
        print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])


if __name__ == "__main__":
    # Works on a single process or across multiple processes with torchrun
    main()

