import torch.nn as nn


def build_projector(
    projector_type: str,
    input_dim: int,
    output_dim: int,
):
    if projector_type == "linear":
        return nn.Linear(input_dim, output_dim)

    if projector_type == "mlp2x_gelu":
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    raise ValueError(
        f"Unsupported projector type: {projector_type}"
    )