from __future__ import annotations

from copy import deepcopy
from typing import Mapping

import torch
import torch.nn as nn


class DeviceAwareStudentRouter(nn.Module):
    """Route known-device samples to specialized students and others globally.

    The router mirrors the device-aware evaluation strategy while keeping model
    construction independent of any particular dataset loader or device label
    encoding.
    """

    def __init__(self, global_model: nn.Module, specialized: Mapping[int, nn.Module]):
        super().__init__()
        self.global_model = global_model
        self.specialized = nn.ModuleDict({str(int(k)): v for k, v in specialized.items()})

    def forward(self, x: torch.Tensor, device_ids: torch.Tensor) -> torch.Tensor:
        if x.size(0) != device_ids.numel():
            raise ValueError("device_ids must contain one entry per input sample")
        output = None
        unique_devices = torch.unique(device_ids.detach()).tolist()
        for dev in unique_devices:
            mask = device_ids == int(dev)
            key = str(int(dev))
            model = self.specialized[key] if key in self.specialized else self.global_model
            logits = model(x[mask])
            if output is None:
                output = logits.new_empty((x.size(0), logits.size(1)))
            output[mask] = logits
        if output is None:
            raise ValueError("empty input batch")
        return output
