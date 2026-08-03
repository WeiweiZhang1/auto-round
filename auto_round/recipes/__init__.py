"""Reusable quantization recipes for production model families."""

from auto_round.recipes.qwen3_omni_mxfp import (
    qwen3_omni_mixed_mxfp_recipe,
    qwen3_omni_mxfp4_experts_recipe,
    qwen3_omni_mxfp8_thinker_recipe,
)

__all__ = [
    "qwen3_omni_mixed_mxfp_recipe",
    "qwen3_omni_mxfp4_experts_recipe",
    "qwen3_omni_mxfp8_thinker_recipe",
]
