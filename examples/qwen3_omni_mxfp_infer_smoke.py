from __future__ import annotations

import argparse

import sglang as sgl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a short SGLang smoke test for a Qwen3-Omni MXFP checkpoint."
    )
    parser.add_argument(
        "--model",
        default="/storage/models/Qwen3-Omni-30B-A3B-Instruct-AutoRound-MXFP4Experts-MXFP8Dense",
    )
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--moe-runner-backend", default="flashinfer_trtllm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = sgl.Engine(
        model_path=args.model,
        trust_remote_code=True,
        mem_fraction_static=0.80,
        context_length=1024,
        disable_cuda_graph=True,
        moe_runner_backend=args.moe_runner_backend,
    )
    try:
        output = engine.generate(
            args.prompt,
            sampling_params={"max_new_tokens": args.max_new_tokens, "temperature": 0},
        )
        print(output)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
