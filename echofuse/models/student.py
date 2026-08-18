from __future__ import annotations

from dataclasses import replace
from typing import Optional

import torch
import torch.nn as nn
from torch.ao.quantization import DeQuantStub, QuantStub
from torchvision.ops import Conv2dNormActivation

from ..config import StudentConfig


def make_divisible(v, divisor=8, min_value=None):
    """Ensure channel counts align to a hardware-friendly divisor."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class GRN(nn.Module):
    """Global Response Normalization with an explicit quant/dequant boundary."""

    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.dequant(x)
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        x = self.gamma * (x * nx) + self.beta + x
        return self.quant(x)


class CPMobileBlock(nn.Module):
    """Expand -> depthwise -> project block used by the compact student."""

    def __init__(self, in_channels, out_channels, expansion_rate, stride):
        super().__init__()
        exp_channels = make_divisible(in_channels * expansion_rate, 8)
        exp_conv = Conv2dNormActivation(
            in_channels,
            exp_channels,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU,
            inplace=False,
        )
        depth_conv = Conv2dNormActivation(
            exp_channels,
            exp_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=exp_channels,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU,
            inplace=False,
        )
        proj_conv = Conv2dNormActivation(
            exp_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=None,
            inplace=False,
        )
        self.after_block_norm = GRN()
        self.after_block_activation = nn.ReLU(inplace=False)
        self.use_shortcut = in_channels == out_channels and stride in (1, (1, 1))
        self.shortcut = nn.Sequential() if self.use_shortcut else None
        self.block = nn.Sequential(exp_conv, depth_conv, proj_conv)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        result = self.block(x)
        if self.use_shortcut:
            result = self.skip_add.add(result, self.shortcut(x))
        result = self.after_block_norm(result)
        result = self.after_block_activation(result)
        return result


class CPMobileStudent(nn.Module):
    """Modified CP-Mobile student used by EchoFuse.

    The original supplied CP-Mobile configuration used a wider 32-channel
    base.  This student keeps the same quantization-ready block design but uses
    the compact configuration in StudentConfig to target the ~60K-parameter,
    ~30M-MAC operating point.
    """

    def __init__(self, config: StudentConfig = StudentConfig()):
        super().__init__()
        self.config = config
        n_classes = config.n_classes
        in_channels = config.in_channels
        base_channels = make_divisible(config.base_channels, 8)
        n_blocks = tuple(config.n_blocks)
        n_stages = len(n_blocks)
        strides = config.stride_dict()

        channels_per_stage = [base_channels] + [
            make_divisible(base_channels * config.channels_multiplier ** stage_id, 8)
            for stage_id in range(n_stages)
        ]

        self.total_block_count = 0
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.in_c = nn.Sequential(
            Conv2dNormActivation(
                in_channels,
                channels_per_stage[0] // 4,
                kernel_size=3,
                stride=2,
                inplace=False,
            ),
            Conv2dNormActivation(
                channels_per_stage[0] // 4,
                channels_per_stage[0],
                activation_layer=nn.ReLU,
                kernel_size=3,
                stride=2,
                inplace=False,
            ),
        )

        self.stages = nn.Sequential()
        for stage_id in range(n_stages):
            stage = self._make_stage(
                channels_per_stage[stage_id],
                channels_per_stage[stage_id + 1],
                n_blocks[stage_id],
                strides=strides,
                expansion_rate=config.expansion_rate,
            )
            self.stages.add_module(f"s{stage_id + 1}", stage)

        self.feed_forward = nn.Sequential(
            nn.Conv2d(
                channels_per_stage[-1],
                n_classes,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(n_classes),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, strides, expansion_rate):
        stage = nn.Sequential()
        for _ in range(n_blocks):
            block_id = self.total_block_count + 1
            bname = f"b{block_id}"
            self.total_block_count += 1
            stride = strides.get(bname, (1, 1))
            block = CPMobileBlock(in_channels, out_channels, expansion_rate, stride)
            stage.add_module(bname, block)
            in_channels = out_channels
        return stage

    def forward_features(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.quant(x)
        x = self.in_c(x)
        stage_features = []
        for stage in self.stages:
            x = stage(x)
            stage_features.append(x)
        return x, stage_features

    def forward(self, x):
        x, _ = self.forward_features(x)
        x = self.feed_forward(x)
        logits = x.squeeze(2).squeeze(2)
        return self.dequant(logits)

    def forward_with_features(self, x):
        x, features = self.forward_features(x)
        logits = self.feed_forward(x).squeeze(2).squeeze(2)
        return self.dequant(logits), features

    def fuse_model(self):
        """Fuse Conv-BN-activation sequences before quantization preparation."""
        for module_name, module_instance in self.named_modules():
            if module_name == "in_c":
                torch.quantization.fuse_modules(module_instance[0], ["0", "1", "2"], inplace=True)
                torch.quantization.fuse_modules(module_instance[1], ["0", "1", "2"], inplace=True)
            elif isinstance(module_instance, CPMobileBlock):
                torch.quantization.fuse_modules(module_instance.block[0], ["0", "1", "2"], inplace=True)
                torch.quantization.fuse_modules(module_instance.block[1], ["0", "1", "2"], inplace=True)
                torch.quantization.fuse_modules(module_instance.block[2], ["0", "1"], inplace=True)
            elif module_name == "feed_forward":
                torch.quantization.fuse_modules(module_instance, ["0", "1"], inplace=True)


def build_cp_mobile_student(
    num_classes: int = 10,
    config: Optional[StudentConfig] = None,
) -> CPMobileStudent:
    config = config or StudentConfig(n_classes=num_classes)
    if config.n_classes != num_classes:
        config = replace(config, n_classes=num_classes)
    return CPMobileStudent(config)
