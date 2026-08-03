from __future__ import annotations

from dataclasses import dataclass


_MXFP4 = {"bits": 4, "group_size": 32, "sym": True, "data_type": "mx_fp"}
_MXFP8 = {"bits": 8, "group_size": 32, "sym": True, "data_type": "mx_fp"}

QWEN3_OMNI_EXPERT_PATTERN = "thinker.model.layers.*.mlp.experts.*"
QWEN3_OMNI_ATTENTION_PATTERN = "thinker.model.layers.*.self_attn.*"
QWEN3_OMNI_BF16_IGNORE = "talker,code2wav"
QWEN3_OMNI_NON_EXPERT_IGNORE = (
    "talker,code2wav,thinker.visual,thinker.audio_tower,"
    "thinker.model.layers.*.self_attn.*,thinker.model.layers.*.mlp.gate,"
    "thinker.lm_head"
)
QWEN3_OMNI_NON_TEXT_IGNORE = (
    "talker,code2wav,thinker.visual,thinker.audio_tower,thinker.lm_head"
)


@dataclass(frozen=True)
class AutoRoundRecipe:
    """Arguments that can be passed directly to AutoRound or the CLI."""

    scheme: str
    layer_config: dict[str, dict]
    ignore_layers: str = QWEN3_OMNI_BF16_IGNORE
    format: str = "llm_compressor"
    model_free: bool = True


def qwen3_omni_mxfp4_experts_recipe() -> AutoRoundRecipe:
    """MXFP4 only for Qwen3-Omni thinker MoE experts.

    Dense thinker attention linears, talker, and code2wav remain BF16. Router
    gates, embeddings, and convolutional layers are skipped by model-free
    AutoRound's existing safety rules.
    """

    return AutoRoundRecipe(
        scheme="MXFP4",
        ignore_layers=QWEN3_OMNI_NON_EXPERT_IGNORE,
        layer_config={
            QWEN3_OMNI_EXPERT_PATTERN: dict(_MXFP4),
            QWEN3_OMNI_ATTENTION_PATTERN: {"bits": 16},
        },
    )


def qwen3_omni_mixed_mxfp_recipe() -> AutoRoundRecipe:
    """Mixed Qwen3-Omni recipe: MXFP4 experts and MXFP8 dense thinker linears."""

    return AutoRoundRecipe(
        scheme="MXFP8",
        ignore_layers=QWEN3_OMNI_NON_TEXT_IGNORE,
        layer_config={
            QWEN3_OMNI_EXPERT_PATTERN: dict(_MXFP4),
        },
    )


def qwen3_omni_mxfp8_thinker_recipe() -> AutoRoundRecipe:
    """MXFP8 W8A8 for Qwen3-Omni thinker transformer linears on Blackwell."""

    return AutoRoundRecipe(
        scheme="MXFP8",
        ignore_layers=QWEN3_OMNI_NON_TEXT_IGNORE,
        layer_config={},
    )
