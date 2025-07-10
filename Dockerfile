FROM python:3.11-slim

# Install git for dataset/tokenizer downloads
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch and related libraries
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu torchvision==0.17.2+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

RUN pip install --no-cache-dir transformers==4.39.2 datasets==2.18.0

# Copy training script
COPY fine_tune_tinyllama.py /workspace/
WORKDIR /workspace

# Default command runs the training script
ENTRYPOINT ["python", "fine_tune_tinyllama.py"]
