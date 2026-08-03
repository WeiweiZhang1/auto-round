from __future__ import annotations

import argparse

from auto_round import AutoRound
from auto_round.recipes import (
    qwen3_omni_mixed_mxfp_recipe,
    qwen3_omni_mxfp4_experts_recipe,
    qwen3_omni_mxfp8_thinker_recipe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize Qwen3-Omni with AutoRound MXFP model-free recipes."
    )
    parser.add_argument(
        "--model",
        default="/storage/models/Qwen3-Omni-30B-A3B-Instruct",
        help="BF16 Qwen3-Omni checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        default="/storage/models/Qwen3-Omni-30B-A3B-Instruct-AutoRound-MXFP4Experts-MXFP8Dense",
        help="Directory for the quantized checkpoint.",
    )
    parser.add_argument(
        "--recipe",
        choices=("mixed", "mxfp4-experts", "mxfp8-thinker"),
        default="mixed",
        help="Quantization recipe to apply.",
    )
    parser.add_argument(
        "--device-map",
        default="0",
        help="AutoRound device_map value. Defaults to GPU 0 for model-free packing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe = (
        qwen3_omni_mixed_mxfp_recipe()
        if args.recipe == "mixed"
        else qwen3_omni_mxfp4_experts_recipe()
        if args.recipe == "mxfp4-experts"
        else qwen3_omni_mxfp8_thinker_recipe()
    )

    AutoRound(
        model=args.model,
        scheme=recipe.scheme,
        layer_config=recipe.layer_config,
        ignore_layers=recipe.ignore_layers,
        model_free=recipe.model_free,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
    ).quantize_and_save(args.output_dir, format=recipe.format)


if __name__ == "__main__":
    main()
