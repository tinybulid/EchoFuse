from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.ensemble import average_teacher_logits, blend_logits, weighted_teacher_logits


def softened_distribution(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return F.softmax(logits / temperature, dim=1)


def student_log_distribution(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return F.log_softmax(logits / temperature, dim=1)


def kl_from_teacher_probability(
    student_logits: torch.Tensor,
    teacher_probability: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return F.kl_div(
        student_log_distribution(student_logits, temperature),
        teacher_probability.detach(),
        reduction="batchmean",
    ) * (temperature ** 2)


def a1_z1_teacher_logits(
    teacher_logits: torch.Tensor,
    z1_weights: torch.Tensor,
    average_weight: float = 0.5,
    z1_weight: float = 0.5,
) -> torch.Tensor:
    """Combine average-teacher logits (a1) and z1 fused logits.

    The source description identifies the a1+z1 combination but does not state
    a distinct coefficient. This implementation therefore exposes both blend
    weights and uses an equal blend by default.
    """
    a1 = average_teacher_logits(teacher_logits)
    z1 = weighted_teacher_logits(teacher_logits, z1_weights)
    return blend_logits((a1, average_weight), (z1, z1_weight))


def a1_z1_teacher_probability(
    teacher_logits: torch.Tensor,
    z1_weights: torch.Tensor,
    temperature: float,
    average_weight: float = 0.5,
    z1_weight: float = 0.5,
) -> torch.Tensor:
    logits = a1_z1_teacher_logits(
        teacher_logits,
        z1_weights,
        average_weight=average_weight,
        z1_weight=z1_weight,
    )
    return softened_distribution(logits, temperature)
