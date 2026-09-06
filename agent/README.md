---
title: SatQuery AI Qwen Controller
sdk: gradio
app_file: app.py
python_version: "3.12"
---

# SatQuery AI Qwen Controller

This Hugging Face Space is the Qwen2.5-7B orchestration service for SatQuery AI. It accepts a user request and one to four short-lived HTTPS signed image URLs. It does not accept uploads, Firebase credentials, local paths, manifests, or public workflow controls.

## Interface

The internal Gradio interface contains exactly `User Request`, `Signed Image URL 1` through `Signed Image URL 4`, `Analyze`, and `Qwen Response`. The backend should create signed URLs and provide any trusted modality, timestamp, channel, and correspondence metadata through the internal request contract.

## Architecture

`schemas.py` defines Pydantic contracts. `planner.py` normalizes semantic intent, checks compatibility, and creates the minimum sufficient `ToolPlan`. `registry.py` describes capabilities without executing them. `executor.py` validates and dispatches calls while keeping artifacts request-local. `services.py` owns authenticated Gradio clients. Specialist adapters live under `qwen/controller/tools/`. `controller.py` integrates evidence and asks Qwen for the final answer.

## Supported workflows

- Single optical or multispectral image: GeoChat.
- Single SAR image: pseudo-RGB preprocessing, then GeoChat.
- Optical plus SAR: GeoChat for each modality, then Qwen synthesis.
- Bi-temporal imagery: M²CD, with semantic GeoChat inspection only when required by the task.
- Dual SAR: M²CD change evidence followed by request-local region preprocessing and GeoChat when semantic interpretation is needed.
- Four URLs: `VV T1`, `VH T1`, `VV T2`, `VH T2`, represented internally as SAR channel/time inputs.

Pseudo-RGB uses robust visualization normalization: `R = VV`, `G = VH`, and `B = (VV + VH) / 2`. It is not a substitute for quantitative SAR values.

## Security and output

`HF_TOKEN` is read only from environment configuration and is used for private specialist Space calls. Signed URLs are opaque, short-lived resources and are never reconstructed or logged. Qwen does not access Firebase. Responses contain task and input summaries, observable execution traces, structured specialist evidence, spatial outputs when supplied by a specialist, an evidence-grounded answer, and explicit failure or uncertainty information. Raw NumPy masks remain request-local and are never serialized into public JSON.

## Configuration

```text
HF_TOKEN=hf_...
GEOCHAT_SPACE=Bireswar26/GeoChat
TEOCHAT_SPACE=Bireswar26/TEOChat
M2CD_SPACE=Bireswar26/M2CD
GEOCHAT_API_NAME=/geochat
TEOCHAT_API_NAME=/teochat
M2CD_API_NAME=/m2cd
```

## Local checks and deployment

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...
python -m compileall .
python -m pytest -q
python app.py
```

The Space needs a CUDA/ZeroGPU-capable runtime for Qwen model generation. Specialist endpoint names and argument schemas remain deployment-specific environment configuration; update their adapters when those external contracts change.
