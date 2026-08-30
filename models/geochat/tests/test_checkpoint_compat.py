from __future__ import annotations

import json

from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download

from models.geochat.src.configuration_geochat import GeoChatConfig
from models.geochat.src.modeling_geochat import GeoChatLlamaForCausalLM


MODEL_ID = "MBZUAI/geochat-7B"


def load_json(filename: str) -> dict:
    """
    Download a JSON file from the GeoChat Hugging Face repository
    and return it as a Python dictionary.
    """
    path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=filename,
    )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 80)
    print("GEOCHAT CHECKPOINT COMPATIBILITY TEST")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load the original GeoChat configuration
    # ------------------------------------------------------------------

    raw_config = load_json("config.json")
    config = GeoChatConfig(**raw_config)

    # ------------------------------------------------------------------
    # Build our architecture without allocating real model weights.
    #
    # This is critical because GeoChat is a 7B model and the local
    # machine only has 8 GB VRAM.
    # ------------------------------------------------------------------

    print("\nBuilding EMPTY architecture...")
    print("(No 7B weights are being allocated.)")

    with init_empty_weights():
        model = GeoChatLlamaForCausalLM(
            config,
            load_vision_weights=False,
        )

    # Get parameter/buffer names from our implementation.
    model_keys = set(model.state_dict().keys())

    # ------------------------------------------------------------------
    # Load the original checkpoint index.
    #
    # The index contains the names of all tensors and tells us which
    # .bin shard contains each tensor.
    # ------------------------------------------------------------------

    index = load_json("pytorch_model.bin.index.json")
    checkpoint_keys = set(index["weight_map"].keys())

    # ------------------------------------------------------------------
    # Compare checkpoint keys against our architecture.
    # ------------------------------------------------------------------

    matching = checkpoint_keys & model_keys
    missing = checkpoint_keys - model_keys
    extra = model_keys - checkpoint_keys

    # ------------------------------------------------------------------
    # Old GeoChat was saved with Transformers 4.31-era LLaMA modules.
    #
    # Those checkpoints contain:
    #
    #   model.layers.X.self_attn.rotary_emb.inv_freq
    #
    # Modern Transformers does not necessarily register these RoPE
    # tensors in the state dict, because the frequencies are derived
    # from the RoPE configuration.
    #
    # Therefore these are expected legacy differences rather than
    # missing learned weights.
    # ------------------------------------------------------------------

    legacy_rotary_keys = {
        key
        for key in missing
        if key.endswith("self_attn.rotary_emb.inv_freq")
    }

    real_missing = missing - legacy_rotary_keys

    # ------------------------------------------------------------------
    # Print results.
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)

    print(f"Checkpoint keys : {len(checkpoint_keys)}")
    print(f"Our model keys  : {len(model_keys)}")
    print(f"Matching        : {len(matching)}")
    print(f"Legacy RoPE keys: {len(legacy_rotary_keys)}")
    print(f"Real missing    : {len(real_missing)}")
    print(f"Extra           : {len(extra)}")

    # ------------------------------------------------------------------
    # Print genuine missing keys.
    # ------------------------------------------------------------------

    if real_missing:
        print("\n--- REAL CHECKPOINT KEYS MISSING FROM OUR MODEL ---")

        for key in sorted(real_missing):
            print(key)

    # ------------------------------------------------------------------
    # Print unexpected keys.
    # ------------------------------------------------------------------

    if extra:
        print("\n--- EXTRA KEYS IN OUR MODEL ---")

        for key in sorted(extra):
            print(key)

    # ------------------------------------------------------------------
    # Report expected legacy RoPE differences.
    # ------------------------------------------------------------------

    if legacy_rotary_keys:
        print(
            "\n✓ Ignoring "
            f"{len(legacy_rotary_keys)} legacy rotary "
            "inv_freq buffers."
        )

    # ------------------------------------------------------------------
    # Final verdict.
    # ------------------------------------------------------------------

    if real_missing or extra:
        raise RuntimeError(
            "GeoChat architecture is not structurally compatible."
        )

    print("\n✓ Architecture is structurally compatible.")


if __name__ == "__main__":
    main()