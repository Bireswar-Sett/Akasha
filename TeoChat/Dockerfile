FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# System packages
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    ca-certificates \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:${PATH}"

# Reproduce the working Python version
RUN conda create -n teochat python=3.9 -y

ENV CONDA_DEFAULT_ENV=teochat
ENV PATH="/opt/conda/envs/teochat/bin:${PATH}"

RUN python --version

# PyTorch environment used by the verified TEOChat setup
RUN pip install --upgrade pip setuptools wheel \
    && pip install \
        torch==2.2.1 \
        torchvision==0.17.1 \
        --index-url https://download.pytorch.org/whl/cu121

# Runtime dependencies required by TEOChat
RUN pip install \
    "numpy<2" \
    transformers==4.31.0 \
    tokenizers==0.13.3 \
    sentencepiece==0.1.99 \
    accelerate==0.26.1 \
    peft==0.7.1 \
    bitsandbytes==0.41.0 \
    timm==0.6.12 \
    einops==0.6.1 \
    einops-exts==0.0.4 \
    scikit-learn==1.2.2 \
    pydantic==2.8.2 \
    httpx==0.27.0 \
    requests==2.32.3 \
    decord==0.6.0 \
    opencv-python-headless \
    pillow \
    fastapi==0.115.2 \
    "uvicorn[standard]==0.32.0" \
    python-multipart==0.0.12 \
    huggingface_hub

# PyTorchVideo is required by the TEOChat code
RUN pip install pytorchvideo

# Compatibility fix used in the working Colab environment
RUN python - <<'PY'
from pathlib import Path

path = Path(
    "/opt/conda/envs/teochat/lib/python3.9/site-packages/"
    "pytorchvideo/transforms/augmentations.py"
)

if path.exists():
    text = path.read_text()
    text = text.replace(
        "import torchvision.transforms.functional_tensor as F_t",
        "import torchvision.transforms.functional as F_t"
    )
    path.write_text(text)
PY

# Copy application source
COPY pyproject.toml .
COPY api.py .
COPY teochat_engine.py .
COPY videollava ./videollava

# Install the TEOChat package itself without re-resolving
# its larger training-time dependency set.
RUN pip install --no-deps .

# Hugging Face cache is expected to be mounted persistently.
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
