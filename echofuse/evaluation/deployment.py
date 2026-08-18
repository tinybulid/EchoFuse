from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parameter_size_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def estimate_conv_linear_macs(model: nn.Module, input_shape=(1, 1, 256, 64), device="cpu") -> int:
    """Approximate MACs from Conv2d and Linear layers using forward hooks."""
    model = deepcopy(model).to(device).eval()
    total = [0]
    hooks = []

    def conv_hook(module, inputs, output):
        out_h, out_w = output.shape[-2:]
        k_h, k_w = module.kernel_size
        per_output = (module.in_channels // module.groups) * k_h * k_w
        total[0] += output.shape[0] * output.shape[1] * out_h * out_w * per_output

    def linear_hook(module, inputs, output):
        batch = output.shape[0] if output.ndim > 1 else 1
        total[0] += batch * module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    with torch.no_grad():
        model(torch.zeros(*input_shape, device=device))
    for hook in hooks:
        hook.remove()
    return int(total[0])


def make_fp16_copy(model: nn.Module) -> nn.Module:
    return deepcopy(model).eval().half()


def save_fp16_state_dict(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fp16_state = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in model.state_dict().items()}
    torch.save(fp16_state, path)


def model_summary(model: nn.Module, input_shape=None) -> dict:
    params = count_parameters(model)
    bytes_ = parameter_size_bytes(model)
    result = {
        "parameters": params,
        "parameter_k": params / 1_000,
        "size_bytes": bytes_,
        "size_kib": bytes_ / 1024,
    }
    if input_shape is not None:
        macs = estimate_conv_linear_macs(model, input_shape=input_shape)
        result["macs"] = macs
        result["mac_m"] = macs / 1_000_000
    return result
