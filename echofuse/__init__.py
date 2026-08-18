"""EchoFuse: compact acoustic-scene classification with learned ensemble KD."""

from .api import build_student, build_z1, build_z2, build_teacher_pool, extract_logmel
from .config import (
    SpectrumConfig,
    StudentConfig,
    OptimizerConfig,
    FusionTrainingConfig,
    KDTrainingConfig,
    TrainingConfig,
)
from .models import Z1FusionNetwork, Z2PerClassFusion, TeacherPool, CPMobileStudent

__all__ = [
    "build_student",
    "build_z1",
    "build_z2",
    "build_teacher_pool",
    "extract_logmel",
    "SpectrumConfig",
    "StudentConfig",
    "OptimizerConfig",
    "FusionTrainingConfig",
    "KDTrainingConfig",
    "TrainingConfig",
    "Z1FusionNetwork",
    "Z2PerClassFusion",
    "TeacherPool",
    "CPMobileStudent",
]

__version__ = "0.1.0"
