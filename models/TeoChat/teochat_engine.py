import torch

from videollava.eval.eval import load_model
from videollava.eval.inference import run_inference_single


class TEOChatEngine:
    """
    Reusable wrapper around the pretrained TEOChat model.
    The model is loaded once and reused for multiple requests.
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

    @torch.inference_mode()
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

        if len(image_paths) != 2:
            raise ValueError("TEOChat requires exactly two optical images: T1 and T2.")

        if not instruction or not instruction.strip():
            raise ValueError("Instruction cannot be empty.")

        prefix = (
            "These are two optical satellite images of the same location at "
            "different times, provided in chronological order T1 then T2: <video>\n"
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
