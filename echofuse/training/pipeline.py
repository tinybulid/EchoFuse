from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch.nn as nn

from ..config import TrainingConfig
from ..models.ensemble import TeacherPool
from ..models.fusion import Z1FusionNetwork, Z2PerClassFusion
from ..models.student import CPMobileStudent
from .engine import choose_device, set_seed
from .fusion import train_fusion_heads
from .kd import train_student_kd


@dataclass
class PipelineArtifacts:
    teacher_pool: TeacherPool
    z1: Z1FusionNetwork
    z2: Z2PerClassFusion
    student: CPMobileStudent
    fusion_history: object
    kd_history: object


def run_reference_pipeline(
    train_batches: Iterable,
    validation_batches: Iterable,
    teachers: Sequence[nn.Module],
    config: TrainingConfig,
) -> PipelineArtifacts:
    """Wire the learned fusion stage and final compact-student KD stage."""
    config.kd.validate()
    set_seed(config.seed)
    device = choose_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if len(teachers) != config.num_teachers:
        raise ValueError(
            f"expected {config.num_teachers} teachers, received {len(teachers)}"
        )

    teacher_pool = TeacherPool(teachers)
    z1 = Z1FusionNetwork(config.num_teachers, config.student)
    z2 = Z2PerClassFusion(
        num_teachers=config.num_teachers,
        num_classes=config.num_classes,
        hidden_dim=config.fusion.z2_hidden_dim,
    )

    fusion_history = train_fusion_heads(
        teacher_pool,
        z1,
        z2,
        train_batches,
        validation_batches,
        device,
        config=config.fusion,
        checkpoint_path=config.output_dir / config.fusion.checkpoint_name,
    )

    student = CPMobileStudent(config.student)
    kd_history = train_student_kd(
        student,
        teacher_pool,
        z1,
        train_batches,
        validation_batches,
        device,
        config=config.kd,
        checkpoint_path=config.output_dir / config.kd.checkpoint_name,
    )

    return PipelineArtifacts(
        teacher_pool=teacher_pool,
        z1=z1,
        z2=z2,
        student=student,
        fusion_history=fusion_history,
        kd_history=kd_history,
    )
