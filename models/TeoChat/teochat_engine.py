
import torch

from videollava.eval.eval import load_model
from videollava.eval.inference import run_inference_single


class TEOChatEngine:
    """
    Lightweight wrapper around the pretrained TEOChat model.

    The model is loaded once and reused for multiple inference requests.
    """

    def __init__(
        self,
        model_path="jirvin16/TEOChat",
        model_base=None,
        device="cuda",
        load_8bit=True,
    ):
        self.device = device

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no GPU is available.")

        print("Loading TEOChat model...")

        (
            self.tokenizer,
            self.model,
            self.processor,
        ) = load_model(
            model_path=model_path,
            model_base=model_base,
            load_8bit=load_8bit,
            device=device,
        )

        print("TEOChat loaded successfully.")

    def analyze(
        self,
        image_paths,
        instruction,
        timestamps=None,
        temperature=0.2,
        max_new_tokens=256,
    ):
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        if not image_paths:
            raise ValueError("At least one image is required.")

        if not instruction or not instruction.strip():
            raise ValueError("Instruction cannot be empty.")

        if len(image_paths) == 1:
            prefix = "This is a satellite image: <video>\n"
        else:
            prefix = (
                "This is a sequence of satellite images capturing the same "
                "location at different times in chronological order: <video>\n"
            )

        prompt = prefix + instruction.strip()

        response = run_inference_single(
            self.model,
            self.processor,
            self.tokenizer,
            prompt,
            image_paths,
            conv_mode="v1",
            timestamps=timestamps or [],
            prompt_strategy="interleave",
            chronological_prefix=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        return response.strip()
