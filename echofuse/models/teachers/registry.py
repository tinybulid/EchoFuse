from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

from .Lenv1 import DSFAmirTeacher
from .LenV2 import DSFMehdy2Teacher
from .LenV2Prime import DSFMehdyTeacher
from .cp_mobile import CPMobileTeacher
from .cp_resnet import CPResNetTeacher


_BUILDERS: Dict[str, Callable[..., nn.Module]] = {
    "len_v1": DSFAmirTeacher,
    "len_v2": DSFMehdy2Teacher,
    "len_v2_prime": DSFMehdyTeacher,
    "cp_mobile": CPMobileTeacher,
    "cp_resnet": CPResNetTeacher,
}


def available_external_teachers():
    return tuple(sorted(_BUILDERS))


def build_external_teacher(name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key not in _BUILDERS:
        raise KeyError(
            f"Unknown external teacher {name!r}. Available: {', '.join(available_external_teachers())}"
        )
    return _BUILDERS[key](num_classes=num_classes, **kwargs)
