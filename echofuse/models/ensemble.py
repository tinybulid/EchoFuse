from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


class TeacherPool(nn.Module):
    """Container for a heterogeneous teacher ensemble."""

    def __init__(self, teachers: Sequence[nn.Module]):
        super().__init__()
        if not teachers:
            raise ValueError("TeacherPool requires at least one teacher")
        self.teachers = nn.ModuleList(list(teachers))

    def __len__(self):
        return len(self.teachers)

    def freeze(self) -> "TeacherPool":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False
        return self

    def unfreeze(self) -> "TeacherPool":
        for parameter in self.parameters():
            parameter.requires_grad = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return stacked logits with shape [B, T, C]."""
        outputs = [teacher(x) for teacher in self.teachers]
        return torch.stack(outputs, dim=1)


def average_teacher_logits(teacher_logits: torch.Tensor) -> torch.Tensor:
    if teacher_logits.ndim != 3:
        raise ValueError("teacher_logits must have shape [B, T, C]")
    return teacher_logits.mean(dim=1)


def weighted_teacher_logits(
    teacher_logits: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if teacher_logits.ndim != 3:
        raise ValueError("teacher_logits must have shape [B, T, C]")
    if weights.ndim != 2:
        raise ValueError("weights must have shape [B, T]")
    if teacher_logits.shape[:2] != weights.shape:
        raise ValueError("teacher logits and mixture weights disagree on [B, T]")
    return (teacher_logits * weights.unsqueeze(-1)).sum(dim=1)


def blend_logits(*items: tuple[torch.Tensor, float]) -> torch.Tensor:
    """Weighted arithmetic blend of logit streams."""
    if not items:
        raise ValueError("at least one logit stream is required")
    total_weight = sum(float(weight) for _, weight in items)
    if total_weight <= 0:
        raise ValueError("blend weights must sum to a positive value")
    result = None
    for logits, weight in items:
        contribution = logits * (float(weight) / total_weight)
        result = contribution if result is None else result + contribution
    return result
