from __future__ import annotations

import json

import gradio as gr
import spaces

from qwen.controller.controller import QwenController
from qwen.controller.executor import ToolExecutor
from qwen.controller.model import QwenEngine
from qwen.controller.schemas import AnalysisRequest
from qwen.controller.services import MAX_CONTROLLER_STEPS, validate_config


validate_config()
controller = QwenController(qwen=QwenEngine(), executor=ToolExecutor(), max_steps=MAX_CONTROLLER_STEPS)


@spaces.GPU(duration=120)
def analyze(user_request: str, url_1: str = "", url_2: str = "", url_3: str = "", url_4: str = "", manifest_json: str = "", physical_files=None) -> str:
    urls = [value.strip() for value in (url_1, url_2, url_3, url_4) if value and value.strip()]
    if physical_files:
        files = physical_files if isinstance(physical_files, list) else [physical_files]
        urls.extend(
            item if isinstance(item, str) else item.get("path", item.get("name", ""))
            for item in files
            if item
        )
    urls = urls[:4]
    try:
        metadata = json.loads(manifest_json) if manifest_json.strip() else {}
        if not isinstance(metadata, dict):
            raise ValueError("Manifest must be a JSON object")
        local_paths = [url for url in urls if not url.startswith(("http://", "https://"))]
        signed_urls = [url for url in urls if url.startswith(("http://", "https://"))]
        request = AnalysisRequest(user_request=user_request, signed_image_urls=signed_urls, local_image_paths=local_paths, metadata=metadata)
        return json.dumps(controller.run_request(request), ensure_ascii=False, indent=2, default=str)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error("Qwen could not complete the requested analysis.") from exc


demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Textbox(label="User Request", lines=4),
        gr.Textbox(label="Signed Image URL 1"),
        gr.Textbox(label="Signed Image URL 2"),
        gr.Textbox(label="Signed Image URL 3"),
        gr.Textbox(label="Signed Image URL 4"),
        gr.Textbox(label="Input Manifest JSON (required for SAR pairing)", lines=8),
        gr.File(label="Direct physical files (up to 4)", file_count="multiple", type="filepath"),
    ],
    outputs=gr.Code(label="Qwen Response", language="json"),
    title="SatQuery AI Qwen Controller",
    api_name="analyze",
)


if __name__ == "__main__":
    demo.launch()
