"""Run `kev.serve` with the backbone's linear layers stored in FP8.

The `kev` provider runs this file in Kev's isolated environment in place of `python -m
kev.serve`, so it imports nothing from this package. FP8 applies to full-weight checkpoints
only: a LoRA adapter cannot be folded into FP8 weights, so loading one is refused.

The weights are stored in FP8 but multiplied in bf16: transformers' FP8 matmul comes from a hub
kernel that needs a newer Triton than the torch Kev pins.
"""

import runpy

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, FineGrainedFP8Config
from transformers.integrations.finegrained_fp8 import FP8Linear

# Side of the square weight blocks `FineGrainedFP8Config` scales by default.
FP8_BLOCK = 128

_load_full_weights = AutoModel.from_pretrained


def load_in_fp8(path, **kwargs):
    # transformers swaps in an FP8 layer even where the weight does not tile into 128-wide
    # blocks, then leaves its scale uninitialized, so those stay in bf16.
    with torch.device("meta"):
        skeleton = AutoModel.from_config(AutoConfig.from_pretrained(path))
    untileable = [
        name
        for name, module in skeleton.named_modules()
        if isinstance(module, torch.nn.Linear)
        and (module.in_features % FP8_BLOCK or module.out_features % FP8_BLOCK)
    ]
    return _load_full_weights(
        path,
        quantization_config=FineGrainedFP8Config(modules_to_not_convert=untileable),
        device_map={"": "cuda"},
        **kwargs,
    )


def bf16_forward(self, input):
    rows, columns = self.block_size
    scale = self.weight_scale_inv.repeat_interleave(rows, 0).repeat_interleave(columns, 1)
    weight = (self.weight.to(torch.float32) * scale).to(input.dtype)
    return F.linear(input, weight, self.bias)


def refuse_adapter(*args, **kwargs):
    raise ValueError("quantization: fp8 needs a full-weight Kev checkpoint, not a LoRA adapter")


AutoModel.from_pretrained = load_in_fp8  # ty: ignore[invalid-assignment]
FP8Linear.forward = bf16_forward
AutoModelForCausalLM.from_pretrained = refuse_adapter  # ty: ignore[invalid-assignment]
runpy.run_module("kev.serve", run_name="__main__", alter_sys=True)
