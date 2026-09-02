from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def main() -> None:
    print("=" * 80)
    print("QWEN MODEL LOADING TEST")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    print(
        f"\nGPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("\nQuantization:")
    print("  load_in_4bit      = True")
    print("  quant type        = nf4")
    print("  double quant      = True")
    print("  compute dtype     = float16")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    print("✓ Tokenizer loaded.")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    print("\nLoading Qwen...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.eval()

    print("✓ Model loaded.")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    print("\nModel type:")
    print(
        f"  {type(model).__name__}"
    )

    print("\nModel device:")
    print(
        f"  {model.device}"
    )

    print("\nModel dtype:")

    dtypes = {
        parameter.dtype
        for parameter in model.parameters()
    }

    for dtype in sorted(
        dtypes,
        key=str,
    ):
        print(
            f"  {dtype}"
        )

    if torch.cuda.is_available():
        print("\nVRAM:")
        print(
            f"  allocated = "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )
        print(
            f"  reserved  = "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        )

    # ------------------------------------------------------------------
    # Tiny sanity check
    # ------------------------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": "What is 2 + 2?",
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    print("\nRunning tiny inference test...")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
        )

    generated_tokens = outputs[
        :,
        inputs["input_ids"].shape[1]:,
    ]

    response = tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=True,
    ).strip()

    print(
        f"\nQwen response: {response!r}"
    )

    if not response:
        raise RuntimeError(
            "Qwen generated an empty response."
        )

    print("\n" + "=" * 80)
    print("✓ QWEN LOAD + SANITY TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()