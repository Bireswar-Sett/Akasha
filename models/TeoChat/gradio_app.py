
import os
from pathlib import Path

import gradio as gr
from teochat_engine import TEOChatEngine


API_KEY = os.environ.get("TEOCHAT_API_KEY")

if not API_KEY:
    raise RuntimeError("TEOCHAT_API_KEY is not set.")


print("Loading TEOChat model...")

engine = TEOChatEngine(
    model_path="jirvin16/TEOChat",
    device="cuda",
    load_8bit=True,
)

print("TEOChat loaded successfully.")


def analyze_with_key(api_key, image_paths, question):
    if api_key != API_KEY:
        raise gr.Error("Invalid TEOChat API key.")

    if not image_paths:
        raise gr.Error("At least one image is required.")

    if not question or not question.strip():
        raise gr.Error("Question cannot be empty.")

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    image_paths = [str(Path(path)) for path in image_paths]

    return engine.analyze(
        image_paths=image_paths,
        instruction=question.strip(),
    )


demo = gr.Interface(
    fn=analyze_with_key,
    inputs=[
        gr.Textbox(label="TEOChat API Key", type="password"),
        gr.File(
            label="Satellite image(s)",
            file_count="multiple",
            file_types=["image"],
            type="filepath",
        ),
        gr.Textbox(
            label="Question",
            placeholder="What can you identify in these images?",
        ),
    ],
    outputs=gr.Textbox(label="TEOChat Response"),
    title="TEOChat Gradio API",
    description="Authenticated TEOChat inference endpoint.",
    api_name="predict",
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
