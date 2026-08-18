from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import StudentConfig
from .ensemble import average_teacher_logits, blend_logits, weighted_teacher_logits
from .student import CPMobileStudent


class Z1FusionNetwork(nn.Module):
    """Sample-adaptive teacher weighting using the student backbone.

    The backbone is the same CP-Mobile student structure; only the final output
    dimension is changed from acoustic classes to one score per teacher.
    """

    def __init__(
        self,
        num_teachers: int = 8,
        student_config: StudentConfig = StudentConfig(),
    ):
        super().__init__()
        if num_teachers <= 0:
            raise ValueError("num_teachers must be positive")
        self.num_teachers = num_teachers
        self.backbone = CPMobileStudent(replace(student_config, n_classes=num_teachers))

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.scores(x), dim=1)

    def fuse(self, x: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        return weighted_teacher_logits(teacher_logits, self(x))


class Z2PerClassFusion(nn.Module):
    """Per-class nonlinear teacher-logit fusion.

    Each target class receives the T logits emitted for that class by the
    teacher pool. A one-hidden-layer MLP maps those T values to one fused logit.
    """

    def __init__(self, num_teachers: int = 8, num_classes: int = 10, hidden_dim: int = 32):
        super().__init__()
        if num_teachers <= 0 or num_classes <= 0 or hidden_dim <= 0:
            raise ValueError("num_teachers, num_classes, and hidden_dim must be positive")
        self.num_teachers = num_teachers
        self.num_classes = num_classes
        self.class_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_teachers, hidden_dim),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_classes)
        ])

    def forward(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        if teacher_logits.ndim != 3:
            raise ValueError("teacher_logits must have shape [B, T, C]")
        b, t, c = teacher_logits.shape
        if t != self.num_teachers or c != self.num_classes:
            raise ValueError(
                f"expected [B, {self.num_teachers}, {self.num_classes}], got {tuple(teacher_logits.shape)}"
            )
        outputs = []
        for class_id, head in enumerate(self.class_heads):
            outputs.append(head(teacher_logits[:, :, class_id]).squeeze(-1))
        return torch.stack(outputs, dim=1)


class LearnedFusion(nn.Module):
    """Convenience wrapper exposing a1, z1, z2, and the a1+z1 KD target."""

    def __init__(
        self,
        z1: Z1FusionNetwork,
        z2: Z2PerClassFusion,
        average_weight: float = 0.5,
        z1_weight: float = 0.5,
    ):
        super().__init__()
        self.z1 = z1
        self.z2 = z2
        self.average_weight = average_weight
        self.z1_weight = z1_weight

    def forward(self, x: torch.Tensor, teacher_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        a1 = average_teacher_logits(teacher_logits)
        z1_weights = self.z1(x)
        z1_logits = weighted_teacher_logits(teacher_logits, z1_weights)
        z2_logits = self.z2(teacher_logits)
        a1z1 = blend_logits(
            (a1, self.average_weight),
            (z1_logits, self.z1_weight),
        )
        return {
            "a1": a1,
            "z1_weights": z1_weights,
            "z1": z1_logits,
            "z2": z2_logits,
            "a1z1": a1z1,
        }
