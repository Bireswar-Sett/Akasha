FROM nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# ------------------------------------------------------------------
# System packages
# ------------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        gcc \
        g++ \
        git \
        libgdal-dev && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# ------------------------------------------------------------------
# Application layout
#
# The source imports:
#
#     from models.geochat...
#
# Therefore we preserve that package hierarchy inside the image.
# ------------------------------------------------------------------

WORKDIR /app

RUN mkdir -p /app/models/geochat

# ------------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------------

COPY requirements.txt /app/models/geochat/requirements.txt

RUN pip install \
    --break-system-packages \
    -r /app/models/geochat/requirements.txt

# ------------------------------------------------------------------
# GeoChat source
# ------------------------------------------------------------------

COPY . /app/models/geochat

# ------------------------------------------------------------------
# Ownership
# ------------------------------------------------------------------

RUN mkdir -p /home/user && \
    chown -R 1000:1000 /home/user && \
    chown -R 1000:1000 /app

USER 1000

ENV HOME=/home/user
ENV HF_HOME=/home/user/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/home/user/.cache/huggingface/hub
ENV PATH=/home/user/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PORT=7860

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "models.geochat.inference.server:app", "--host", "0.0.0.0", "--port", "7860"]