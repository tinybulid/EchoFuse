from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .config import SpectrumConfig, StudentConfig
from .models.ensemble import TeacherPool
from .models.fusion import Z1FusionNetwork, Z2PerClassFusion
from .models.student import CPMobileStudent
from .models.teachers import build_external_teacher
from .spectrum import audio_file_to_logmel, waveform_to_logmel


def build_student(num_classes: int = 10, config: Optional[StudentConfig] = None) -> CPMobileStudent:
    if config is None:
        config = StudentConfig(n_classes=num_classes)
    return CPMobileStudent(config)


def build_z1(num_teachers: int = 8, student_config: StudentConfig = StudentConfig()) -> Z1FusionNetwork:
    return Z1FusionNetwork(num_teachers=num_teachers, student_config=student_config)


def build_z2(num_teachers: int = 8, num_classes: int = 10, hidden_dim: int = 32) -> Z2PerClassFusion:
    return Z2PerClassFusion(num_teachers=num_teachers, num_classes=num_classes, hidden_dim=hidden_dim)


def build_teacher_pool(teachers: Sequence[nn.Module]) -> TeacherPool:
    return TeacherPool(teachers)


def extract_logmel(source, source_rate=None, target_frames=None, config: SpectrumConfig = SpectrumConfig()):
    if torch.is_tensor(source):
        if source_rate is None:
            raise ValueError("source_rate is required when source is a waveform tensor")
        return waveform_to_logmel(source, source_rate=source_rate, config=config, target_frames=target_frames)
    return audio_file_to_logmel(Path(source), config=config, target_frames=target_frames)
