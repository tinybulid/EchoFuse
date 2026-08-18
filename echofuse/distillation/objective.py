from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .logits import kl_from_teacher_probability


@dataclass
class KDLossBreakdown:
    total: torch.Tensor
    label: torch.Tensor
    distillation: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "label": float(self.label.detach().cpu()),
            "distillation": float(self.distillation.detach().cpu()),
        }


def kd_objective(
    student_logits: torch.Tensor,
    teacher_probability: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    temperature: float,
) -> KDLossBreakdown:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    ce = F.cross_entropy(student_logits, labels)
    kd = kl_from_teacher_probability(student_logits, teacher_probability, temperature)
    total = (1.0 - alpha) * ce + alpha * kd
    return KDLossBreakdown(total=total, label=ce, distillation=kd)
