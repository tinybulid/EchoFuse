from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SpectrumConfig:
    """Log-Mel front-end settings retained from the previous package."""

    sample_rate: int = 32_000
    n_fft: int = 4_096
    hop_length: int = 502
    n_mels: int = 256
    power: float = 2.0
    top_db: float = 80.0
    center: bool = True
    normalized: bool = False


@dataclass(frozen=True)
class StudentConfig:
    """Modified CP-Mobile student sized for the ~60K / ~30M regime.

    For a [1, 1, 256, 64] input, this reference configuration is roughly
    61K parameters and 30M convolution MACs.  It keeps the CP-Mobile
    expand-depthwise-project design while reducing the original width and
    changing the block schedule/downsampling pattern.
    """

    n_classes: int = 10
    in_channels: int = 1
    base_channels: int = 24
    channels_multiplier: float = 2.0
    expansion_rate: float = 2.5
    n_blocks: tuple[int, ...] = (2, 3, 1)
    # Tuple representation keeps the dataclass immutable/hashable.
    strides: tuple[tuple[str, tuple[int, int]], ...] = (("b4", (2, 2)),)

    def stride_dict(self) -> dict[str, tuple[int, int]]:
        return dict(self.strides)


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size_reference: int = 64
    scheduler: str = "cosine"


@dataclass(frozen=True)
class FusionTrainingConfig:
    """Joint z1/z2 fusion training configuration."""

    epochs: int = 80
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    z1_loss_weight: float = 0.5
    z2_loss_weight: float = 0.5
    z2_hidden_dim: int = 32
    train_teachers: bool = True
    early_stopping_patience: int = 15
    checkpoint_name: str = "tau_fusion_best.pth"


@dataclass(frozen=True)
class KDTrainingConfig:
    """Hard-label + temperature-scaled ensemble KD configuration.

    The supplied method specifies the objective but does not fix alpha or T in
    the provided text.  They are therefore intentionally explicit inputs here
    instead of being silently presented as source-derived constants.
    """

    epochs: int = 80
    alpha: Optional[float] = None
    temperature: Optional[float] = None
    average_teacher_weight: float = 0.5
    z1_teacher_weight: float = 0.5
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    early_stopping_patience: int = 15
    checkpoint_name: str = "tau_student_kd_best.pth"

    def validate(self) -> "KDTrainingConfig":
        if self.alpha is None:
            raise ValueError("KD alpha must be supplied explicitly")
        if self.temperature is None:
            raise ValueError("KD temperature must be supplied explicitly")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.average_teacher_weight < 0 or self.z1_teacher_weight < 0:
            raise ValueError("ensemble blend weights must be non-negative")
        if self.average_teacher_weight + self.z1_teacher_weight <= 0:
            raise ValueError("at least one ensemble blend weight must be positive")
        return self


@dataclass(frozen=True)
class TrainingConfig:
    num_classes: int = 10
    num_teachers: int = 8
    seed: int = 42
    device: Optional[str] = None
    output_dir: Path = Path("checkpoints")
    student: StudentConfig = field(default_factory=StudentConfig)
    fusion: FusionTrainingConfig = field(default_factory=FusionTrainingConfig)
    kd: KDTrainingConfig = field(default_factory=KDTrainingConfig)
