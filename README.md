# Cluster Labs

![Cluster](docs/Cluster.gif)

This repository contains four labs for getting comfortable with a Slurm based cluster and experimenting with distributed language model training.  The material and code were prepared by Assistant Lecturer: **Eng. Ahmed Métwalli** ([ResearchGate](https://www.researchgate.net/profile/Ahmed-Metwalli), [LinkedIn](https://www.linkedin.com/in/ahmed-m%C3%A9twalli/)), affiliated with the **AASTMT Alamein Campus College of Artificial Intelligence**.

## Lab Overview

1. **Lab 0 – Getting Started**
   - Walks you through connecting to the cluster via FortiClient SSL‑VPN and an SFTP workflow inside VS Code.
   - Introduces basic Slurm commands and provides a 30 second smoke test script.
2. **Lab 1 – Simple Model**
   - Data parallel text classification example located in `labs/simple_model`.
3. **Lab 2 – Fine Tuning**
   - Fine tune a causal language model using HuggingFace trainer (`labs/fine_tuning`).
4. **Lab 3 – Transfer & RAG**
   - Covers transfer learning (`labs/transfer_learning`) and a small retrieval‑augmented generation demo (`labs/ragging`).
5. **Lab 4 – Tiny Model**
   - Minimal BERT classifier under 1 M parameters in `labs/tiny`.


Refer to [docs/overview.md](docs/overview.md) for a detailed tour of the code.

## Learning Outcomes

After completing these labs you will be able to:

- Establish a secure SFTP workflow with VS Code.
- Create and manage isolated Python environments on the cluster.
- Submit, monitor and cancel jobs through Slurm.
- Experiment with distributed training, transfer learning and RAG pipelines.

## Utilities

All helper scripts live inside the `utils` directory:

- `utils/Connect2Cluster.py` – Paramiko based interactive SSH client that keeps your terminal responsive while forwarding Ctrl‑C and resize events.
- `utils/checker.py` – Collect information about the cluster: `sinfo`, `scontrol`, system limits and loaded modules.
- `utils/parallel_utils.py` – Thin wrappers around `torch.distributed` initialization used by all training scripts.

Run these tools with `python utils/<tool>.py`.

## Getting Started (Lab 0)

1. **Connect through FortiClient VPN**
   1. Install the VPN‑only edition from Fortinet.
   2. Create a new *SSL‑VPN* connection targeting `https://sslvpn.aast.edu:443/HPCGrid`.
   3. Save your campus credentials and connect.
2. **VS Code SFTP Workflow**
   1. Install the "SFTP" extension by *liximomo*.
   2. Add `.vscode/sftp.json` similar to:
      ```json
      {
        "name": "Mito-EntryPoint",
        "protocol": "sftp",
        "host": "10.1.8.4",
        "port": 22,
        "username": "<your username>",
        "password": "<password>",
        "remotePath": "/home/<user>/project",
        "uploadOnSave": false,
        "syncMode": "update"
      }
      ```
   3. Launch `python utils/Connect2Cluster.py` or choose *SFTP → Open SSH in Terminal* to log in.
3. **Environment Setup**
   1. After logging in, enable the system Python and create a fresh virtual environment:
      ```bash
      source /opt/bin/llamaenv/activate
      cd ~
      python3.11 -m venv llamaenv_local
      ```
   2. Activate the environment and install the course requirements (upload `requirements.txt` first):
      ```bash
      source ~/llamaenv_local/bin/activate
      pip install -r requirements.txt
      ```
   3. Reactivate the environment whenever you start a terminal session:
      ```bash
      source ~/llamaenv_local/bin/activate
      ```
      Consider adding this command to `~/.bashrc` or defining an alias for convenience.
4. **Smoke Test**
   Save the following as `smoke.sbatch` and submit with `sbatch smoke.sbatch`:
   ```bash
   #!/bin/bash
   #SBATCH -J smoke_test
   #SBATCH -o smoke_%j.out
   #SBATCH -e smoke_%j.err
   #SBATCH -N 1 -n 1
   #SBATCH -t 00:00:30

   echo "Node: $(hostname)"
   echo "Start: $(date)"
   sleep 5
   echo "End  : $(date)"
   ```
   Monitor jobs with `squeue -u $USER` and cancel with `scancel <jobID>` if needed.

## Typical Slurm Commands

```
# Partition status
sinfo -R -o "%P %.6t %.6D %.15f"
# Your jobs with reason
squeue -u $USER -o "%.9i %.2t %.10M %.15R"
# Cancel a job
scancel <jobID>
```

Remember to check that no jobs are running before logging out and sync any results from `/scratch` back to your workstation.

## Tiny Model Example

`labs/tiny` contains a lightweight BERT text classifier (<1 M parameters).  The code illustrates distributed training with `torchrun` and includes a small testing script.

### Training

1. Optionally pre-cache the dataset and tokenizer:
   ```bash
   python - <<'PY'
   import os, datasets, transformers
   os.environ["HF_HOME"] = os.path.expanduser("~/.cache/hf_tiny")
   datasets.load_dataset("ag_news", split="train[:2000]",
                         cache_dir=f"{os.environ['HF_HOME']}/ds")
   transformers.AutoTokenizer.from_pretrained(
       "google/bert_uncased_L-2_H-128_A-2",
       cache_dir=f"{os.environ['HF_HOME']}/tok")
   PY
   ```
2. Request a node and launch training:
   ```bash
   salloc -N1 -n1 -c2 -p parallel --time=00:05:00 --exclusive

   source ~/llamaenv_local/bin/activate
   export HF_HOME=$HOME/.cache/hf_tiny
   export MASTER_ADDR=$(hostname)
   export MASTER_PORT=$((20000 + RANDOM % 10000))

   torchrun --nproc_per_node=2 \
     --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
     labs/tiny/train_tiny.py --subset 2000 --epochs 3
   ```
   Checkpoints are written to the `tiny_out` directory.

### Testing

Verify the trained model using `test_tiny.py`:

```bash
export HF_HOME=$HOME/.cache/hf_tiny
python labs/tiny/test_tiny.py --ckpt tiny_out
```

The script reports accuracy on a 512-example slice and prints a few sample predictions.
