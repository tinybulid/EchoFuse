from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn

from ..batches import move_batch, parse_batch
from ..config import KDTrainingConfig
from ..distillation.logits import a1_z1_teacher_probability
from ..distillation.objective import kd_objective
from ..models.ensemble import TeacherPool
from ..models.fusion import Z1FusionNetwork
from .engine import (
    EarlyStopping,
    EpochStats,
    TrainingHistory,
    make_cosine_scheduler,
    make_optimizer,
    maybe_save_best,
    simple_validation,
)


def _freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def train_student_kd(
    student: nn.Module,
    teacher_pool: TeacherPool,
    z1: Z1FusionNetwork,
    train_batches: Iterable,
    validation_batches: Iterable,
    device: str | torch.device,
    config: KDTrainingConfig,
    checkpoint_path: str | Path | None = None,
    augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> TrainingHistory:
    """Simple KD training loop matching the final compact-student objective.

    The teacher target is formed from the average teacher logits (a1) and the
    sample-adaptive z1 output.  z2 remains available for ensemble analysis but
    is not part of this final student target.
    """
    config.validate()
    device = torch.device(device)
    student = student.to(device)
    teacher_pool = _freeze(teacher_pool.to(device))
    z1 = _freeze(z1.to(device))

    optimizer = make_optimizer(
        student,
        config.optimizer.learning_rate,
        config.optimizer.weight_decay,
    )
    scheduler = make_cosine_scheduler(optimizer, config.epochs)
    checkpoint_path = checkpoint_path or config.checkpoint_name
    stopper = EarlyStopping(config.early_stopping_patience)
    history = TrainingHistory()

    for epoch in range(config.epochs):
        student.train()
        loss_sum = 0.0
        correct = 0
        samples = 0
        piece_sums = {"label": 0.0, "distillation": 0.0}

        for raw in train_batches:
            batch = move_batch(parse_batch(raw), device)
            x = augment(batch.inputs) if augment is not None else batch.inputs

            with torch.no_grad():
                teacher_logits = teacher_pool(x)
                z1_weights = z1(x)
                teacher_probability = a1_z1_teacher_probability(
                    teacher_logits,
                    z1_weights,
                    temperature=float(config.temperature),
                    average_weight=config.average_teacher_weight,
                    z1_weight=config.z1_teacher_weight,
                )

            student_logits = student(x)
            breakdown = kd_objective(
                student_logits=student_logits,
                teacher_probability=teacher_probability,
                labels=batch.labels,
                alpha=float(config.alpha),
                temperature=float(config.temperature),
            )

            optimizer.zero_grad(set_to_none=True)
            breakdown.total.backward()
            optimizer.step()

            n = batch.labels.numel()
            loss_sum += float(breakdown.total.detach().cpu()) * n
            correct += int((student_logits.argmax(1) == batch.labels).sum().item())
            samples += n
            piece_sums["label"] += float(breakdown.label.detach().cpu()) * n
            piece_sums["distillation"] += float(breakdown.distillation.detach().cpu()) * n

        scheduler.step()
        if samples == 0:
            raise ValueError("training iterable produced no samples")

        train_stats = EpochStats(
            loss=loss_sum / samples,
            accuracy=correct / samples,
            samples=samples,
            pieces={key: value / samples for key, value in piece_sums.items()},
        )
        val_stats = simple_validation(student, validation_batches, device)
        history.train.append(train_stats)
        history.validation.append(val_stats)

        improved = maybe_save_best(
            history,
            val_stats,
            epoch,
            checkpoint_path,
            student,
            optimizer,
            scheduler,
            config,
            extra={
                "stage": "kd",
                "teacher_count": len(teacher_pool),
                "teacher_target": "a1+z1",
            },
        )
        if improved:
            stopper.best = val_stats.accuracy
            stopper.bad_epochs = 0
        elif stopper.update(val_stats.accuracy):
            break

    return history
