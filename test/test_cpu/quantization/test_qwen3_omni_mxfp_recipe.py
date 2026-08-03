import pytest

from auto_round.compressors.model_free import (
    _build_mxfp_quantization_config,
    _PatternMatcher,
    get_predefined_ignore_layers_from_config,
)
from auto_round.recipes import (
    qwen3_omni_mixed_mxfp_recipe,
    qwen3_omni_mxfp4_experts_recipe,
    qwen3_omni_mxfp8_thinker_recipe,
)

from ...envs import require_compressed_tensors


_QWEN3_OMNI_CFG = {
    "architectures": ["Qwen3OmniMoeThinkerForConditionalGeneration"],
    "model_type": "qwen3_omni_moe",
}


@require_compressed_tensors
def test_qwen3_omni_mixed_recipe_config_contract():
    recipe = qwen3_omni_mixed_mxfp_recipe()
    default = {"bits": 8, "group_size": 32, "sym": True, "data_type": "mx_fp"}
    quantized = [
        "thinker.model.layers.0.self_attn.q_proj",
        "thinker.model.layers.0.mlp.experts.0.gate_proj",
        "thinker.model.layers.0.mlp.experts.0.down_proj",
    ]
    cfg = _build_mxfp_quantization_config(
        default,
        quantized,
        ignored_layers=["talker.model.layers.0.self_attn.q_proj", "code2wav.foo"],
        layer_config=recipe.layer_config,
    )

    assert cfg["provider"] == "auto-round"
    assert set(cfg["ignore"]) == {
        "talker.model.layers.0.self_attn.q_proj",
        "code2wav.foo",
    }
    mxfp4_group = next(
        g
        for g in cfg["config_groups"].values()
        if g["format"] == "mxfp4-pack-quantized"
    )
    mxfp8_group = next(
        g for g in cfg["config_groups"].values() if g["format"] == "mxfp8-quantized"
    )
    assert mxfp4_group["targets"][0] == "RoutedExperts"
    assert "thinker.model.layers.0.mlp.experts.0.gate_proj" in mxfp4_group["targets"]
    assert mxfp8_group["targets"] == ["Linear"]


def test_qwen3_omni_predefined_ignore_keeps_talker_and_code2wav_bf16():
    ignored = get_predefined_ignore_layers_from_config(_QWEN3_OMNI_CFG)
    assert "talker" in ignored
    assert "code2wav" in ignored


def test_qwen3_omni_mxfp4_experts_recipe_keeps_attention_bf16():
    recipe = qwen3_omni_mxfp4_experts_recipe()
    matcher = _PatternMatcher(
        ignore_patterns=[],
        layer_config=recipe.layer_config,
        default_scheme={"bits": 4, "group_size": 32, "sym": True, "data_type": "mx_fp"},
    )

    assert matcher.resolve_scheme("thinker.model.layers.0.self_attn.q_proj.weight") is None
    assert (
        matcher.resolve_scheme(
            "thinker.model.layers.0.mlp.experts.0.gate_proj.weight"
        )["bits"]
        == 4
    )


def test_qwen3_omni_mxfp8_thinker_recipe_keeps_non_text_stages_bf16():
    recipe = qwen3_omni_mxfp8_thinker_recipe()

    assert recipe.scheme == "MXFP8"
    assert "talker" in recipe.ignore_layers
    assert "code2wav" in recipe.ignore_layers
    assert "thinker.visual" in recipe.ignore_layers
    assert "thinker.audio_tower" in recipe.ignore_layers


@require_compressed_tensors
def test_qwen3_omni_mxfp4_experts_recipe_uses_explicit_expert_targets():
    recipe = qwen3_omni_mxfp4_experts_recipe()
    cfg = _build_mxfp_quantization_config(
        {"bits": 4, "group_size": 32, "sym": True, "data_type": "mx_fp"},
        ["thinker.model.layers.0.mlp.experts.0.gate_proj"],
        ignored_layers=["thinker.model.layers.0.self_attn.q_proj", "talker"],
        layer_config=recipe.layer_config,
    )

    group = cfg["config_groups"]["group_0"]
    assert cfg["format"] == "mxfp4-pack-quantized"
    assert "RoutedExperts" in group["targets"]
    assert "Linear" not in group["targets"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
