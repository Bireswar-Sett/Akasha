from transformers import LlamaConfig


class GeoChatConfig(LlamaConfig):
    """
    Configuration for the GeoChat multimodal LLaMA model.

    Extends LlamaConfig with the vision and multimodal settings
    required by GeoChat.
    """

    model_type = "geochat"

    def __init__(
        self,
        mm_hidden_size=1024,
        mm_projector_type="mlp2x_gelu",
        mm_vision_tower="openai/clip-vit-large-patch14-336",
        mm_vision_select_layer=-2,
        mm_vision_select_feature="patch",
        mm_use_im_patch_token=False,
        mm_use_im_start_end=False,
        mm_resampler_type=None,
        mm_resampler_num_latents=None,
        image_aspect_ratio="pad",
        image_grid_pinpoints=None,
        freeze_mm_mlp_adapter=False,
        freeze_mm_vision_resampler=False,
        tune_mm_mlp_adapter=False,
        tune_mm_vision_resampler=False,
        unfreeze_mm_vision_tower=False,
        use_mm_proj=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Vision tower
        self.mm_vision_tower = mm_vision_tower
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature

        # Multimodal projector
        self.mm_hidden_size = mm_hidden_size
        self.mm_projector_type = mm_projector_type
        self.use_mm_proj = use_mm_proj

        # Image token configuration
        self.mm_use_im_patch_token = mm_use_im_patch_token
        self.mm_use_im_start_end = mm_use_im_start_end

        # Optional vision resampler
        self.mm_resampler_type = mm_resampler_type
        self.mm_resampler_num_latents = mm_resampler_num_latents

        # Image preprocessing configuration
        self.image_aspect_ratio = image_aspect_ratio
        self.image_grid_pinpoints = image_grid_pinpoints

        # Training controls
        self.freeze_mm_mlp_adapter = freeze_mm_mlp_adapter
        self.freeze_mm_vision_resampler = freeze_mm_vision_resampler
        self.tune_mm_mlp_adapter = tune_mm_mlp_adapter
        self.tune_mm_vision_resampler = tune_mm_vision_resampler
        self.unfreeze_mm_vision_tower = unfreeze_mm_vision_tower