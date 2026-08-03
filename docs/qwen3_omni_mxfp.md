# Qwen3-Omni AutoRound MXFP

Qwen3-Omni requires a stage-aware MXFP recipe. The thinker can be quantized, but
speech-generation stages must remain BF16.

```bash
python examples/qwen3_omni_mxfp_quantize.py \
  --model /storage/models/Qwen3-Omni-30B-A3B-Instruct \
  --output-dir /storage/models/Qwen3-Omni-30B-A3B-Instruct-AutoRound-MXFP4Experts-MXFP8Dense
```

The mixed recipe exports compressed-tensors with `provider: auto-round`, MXFP4
for thinker MoE experts, MXFP8 for dense thinker linears, and BF16
`talker`/`code2wav`.

For experts-only MXFP4, use:

```python
from auto_round.recipes import qwen3_omni_mxfp4_experts_recipe
```

For an MXFP8-only B200 smoke checkpoint, pass
`--recipe mxfp8-thinker --output-dir /storage/models/Qwen3-Omni-30B-A3B-Instruct-AutoRound-MXFP8Thinker`.

Runtime constraints:

- MXFP4 is supported only for thinker MoE experts.
- MXFP8 dense W8A8 execution requires Blackwell SM100/SM120 CUDA GPUs.
- Quantized `talker` and `code2wav` checkpoints are rejected by SGLang-Omni.

After installing SGLang with Qwen3-Omni MXFP support, run a short text smoke
test:

```bash
PYTHONPATH=/path/to/sglang/python python examples/qwen3_omni_mxfp_infer_smoke.py \
  --model /storage/models/Qwen3-Omni-30B-A3B-Instruct-AutoRound-MXFP4Experts-MXFP8Dense
```
