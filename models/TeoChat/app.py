from __future__ import annotations

from pathlib import Path

import gradio as gr

from response_adapter import normalize_response
from service_contract import require_temporal_pair
from teochat_engine import TEOChatEngine


engine = TEOChatEngine(
    model_path="jirvin16/TEOChat",
    device="cuda",
    load_8bit=True,
)


def analyze_specialist(
    image_t1: str | None,
    image_t2: str | None,
    prompt: str,
    max_new_tokens: int = 256,
) -> dict:
    try:
        temporal_pair = require_temporal_pair(image_t1, image_t2)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    if not prompt or not prompt.strip():
        raise gr.Error("Specialist prompt cannot be empty.")
    try:
        response = engine.analyze(
            image_paths=[str(Path(path)) for path in temporal_pair],
            instruction=prompt.strip(),
            max_new_tokens=int(max_new_tokens),
        )
        return normalize_response(response, len(temporal_pair))
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error("TEOChat inference failed.") from exc


demo = gr.Interface(
    fn=analyze_specialist,
    inputs=[
        gr.File(label="Image T1", type="filepath"),
        gr.File(label="Image T2", type="filepath"),
        gr.Textbox(label="Specialist Prompt", lines=4),
        gr.Number(value=256, minimum=1, maximum=1024, precision=0, visible=False),
    ],
    outputs=gr.JSON(label="TEOChat Evidence"),
    title="TEOChat Specialist",
    api_name="teochat",
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
