import json

from huggingface_hub import hf_hub_download

from src.configuration_geochat import GeoChatConfig


MODEL_ID = "MBZUAI/geochat-7B"


def test_geochat_config():
    path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="config.json",
    )

    with open(path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)

    config = GeoChatConfig(**raw_config)

    assert config.model_type == "geochat"
    assert config.hidden_size == 4096
    assert config.num_hidden_layers == 32
    assert config.num_attention_heads == 32

    assert config.mm_hidden_size == 1024
    assert config.mm_projector_type == "mlp2x_gelu"
    assert (
        config.mm_vision_tower
        == "openai/clip-vit-large-patch14-336"
    )

    print("GeoChatConfig loaded successfully.")