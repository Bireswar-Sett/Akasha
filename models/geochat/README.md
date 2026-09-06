---
title: Akasha GeoChat
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# GeoChat Specialist

GeoChat is a one-image remote-sensing specialist called by the Qwen
orchestrator. The model and processor remain unchanged; Qwen supplies the
task-specific prompt and performs final evidence synthesis.

## Service contract

The Qwen-compatible Gradio endpoint at the Space root is:

```text
/geochat
```

It accepts one image, one specialist prompt, and an internal generation limit:

```text
image: uploaded file or signed-URL file reference
prompt: task-specific specialist instruction
max_new_tokens: 1..512
```

The endpoint returns JSON containing `status`, `tool`, `response`, `answer`,
`evidence`, `model`, and `modality`. Text returned by the model is preserved as
observation evidence; confidence and spatial outputs remain empty unless the
model actually produces them.

Bounding boxes are handled through the Qwen-generated prompt, for example:

```text
Analyze only the supplied region {20,30,70,80|15}.
```

GeoChat does not parse, invent, or modify bbox geometry and does not access
Firebase. Multiple image comparisons use multiple independent one-image calls.

The existing HTTP compatibility routes `/http/geochat`, `/http/geochat/sar`,
and `/health` remain available. The model is loaded once at service startup.